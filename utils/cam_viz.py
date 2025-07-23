import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL
import imageio
from easydict import EasyDict as edict
import plotly.graph_objects as go
import evo.core.geometry as geometry

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

class Quaternion():

    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R

    def R_to_q(self,R,eps=1e-8): # [B,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # FIXME: this function seems a bit problematic, need to double-check
        row0,row1,row2 = R.unbind(dim=-2)
        R00,R01,R02 = row0.unbind(dim=-1)
        R10,R11,R12 = row1.unbind(dim=-1)
        R20,R21,R22 = row2.unbind(dim=-1)
        t = R[...,0,0]+R[...,1,1]+R[...,2,2]
        r = (1+t+eps).sqrt()
        qa = 0.5*r
        qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
        qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
        qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
        q = torch.stack([qa,qb,qc,qd],dim=-1)
        for i,qi in enumerate(q):
            if torch.isnan(qi).any():
                K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
                                 torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R02],dim=-1),
                                 torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
                                 torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
                K = K[i]
                eigval,eigvec = torch.linalg.eigh(K)
                V = eigvec[:,eigval.argmax()]
                q[i] = torch.stack([V[3],V[0],V[1],V[2]])
        return q

    def invert(self,q):
        qa,qb,qc,qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1,keepdim=True)
        q_inv = torch.stack([qa,-qb,-qc,-qd],dim=-1)/norm**2
        return q_inv

    def product(self,q1,q2): # [B,4]
        q1a,q1b,q1c,q1d = q1.unbind(dim=-1)
        q2a,q2b,q2c,q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack([q1a*q2a-q1b*q2b-q1c*q2c-q1d*q2d,
                                  q1a*q2b+q1b*q2a+q1c*q2d-q1d*q2c,
                                  q1a*q2c-q1b*q2d+q1c*q2a+q1d*q2b,
                                  q1a*q2d+q1b*q2c-q1c*q2b+q1d*q2a],dim=-1)
        return hamil_prod

pose = Pose()
lie = Lie()
quaternion = Quaternion()

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)
def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)
def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)
def cam2world(X,pose):
    X_hom = to_hom(X)
    #pose_inv = Pose().invert(pose)
    pose_inv = pose
    return X_hom@pose_inv.transpose(-1,-2)

def angle_to_rotation_matrix(a,axis):
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(),-a.sin(),O],dim=-1),
                     torch.stack([a.sin(),a.cos(),O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M

def get_center_and_ray(opt,pose,intr=None): # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    assert(opt.camera.model=="perspective")
    with torch.no_grad():
        # compute image coordinate grid
        y_range = torch.arange(opt.H,dtype=torch.float32,device=opt.device).add_(0.5)
        x_range = torch.arange(opt.W,dtype=torch.float32,device=opt.device).add_(0.5)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    # compute center and ray
    batch_size = len(pose)
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose) # [B,HW,3]
    center_3D = cam2world(center_3D,pose) # [B,HW,3]
    ray = grid_3D-center_3D # [B,HW,3]
    return center_3D,ray

def get_3D_points_from_depth(opt,center,ray,depth,multi_samples=False):
    if multi_samples: center,ray = center[:,:,None],ray[:,:,None]
    # x = c+dv
    points_3D = center+ray*depth # [B,HW,3]/[B,HW,N,3]/[N,3]
    return points_3D

def convert_NDC(opt,center,ray,intr,near=1):
    # shift camera center (ray origins) to near plane (z=1)
    # (unlike conventional NDC, we assume the cameras are facing towards the +z direction)
    center = center+(near-center[...,2:])/ray[...,2:]*ray
    # projection
    cx,cy,cz = center.unbind(dim=-1) # [B,HW]
    rx,ry,rz = ray.unbind(dim=-1) # [B,HW]
    scale_x = intr[:,0,0]/intr[:,0,2] # [B]
    scale_y = intr[:,1,1]/intr[:,1,2] # [B]
    cnx = scale_x[:,None]*(cx/cz)
    cny = scale_y[:,None]*(cy/cz)
    cnz = 1-2*near/cz
    rnx = scale_x[:,None]*(rx/rz-cx/cz)
    rny = scale_y[:,None]*(ry/rz-cy/cz)
    rnz = 2*near/cz
    center_ndc = torch.stack([cnx,cny,cnz],dim=-1) # [B,HW,3]
    ray_ndc = torch.stack([rnx,rny,rnz],dim=-1) # [B,HW,3]
    return center_ndc,ray_ndc

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def procrustes_analysis(X0,X1): # [N,3]
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def get_novel_view_poses(opt,pose_anchor,N=60,scale=1):
    # create circular viewpoints (small oscillations)
    theta = torch.arange(N)/N*2*np.pi
    R_x = angle_to_rotation_matrix((theta.sin()*0.05).asin(),"X")
    R_y = angle_to_rotation_matrix((theta.cos()*0.05).asin(),"Y")
    pose_rot = pose(R=R_y@R_x)
    pose_shift = pose(t=[0,0,-4*scale])
    pose_shift2 = pose(t=[0,0,3.8*scale])
    pose_oscil = pose.compose([pose_shift,pose_rot,pose_shift2])
    pose_novel = pose.compose([pose_oscil,pose_anchor.cpu()[None]])
    return pose_novel


@torch.no_grad()
def tb_image(opt,tb,step,group,name,images,num_vis=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images[:,0].cpu(),cmap=cmap)
    return images

def dump_images(opt,idx,name,images,masks=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap) # [B,3,H,W]
    images = images.cpu().permute(0,2,3,1).numpy() # [B,H,W,3]
    for i,img in zip(idx,images):
        fname = "{}/dump/{}_{}.png".format(opt.output_path,i,name)
        img_uint8 = (img*255).astype(np.uint8)
        imageio.imsave(fname,img_uint8)

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

def color_border(images,colors,width=3):
    images_pad = []
    for i,image in enumerate(images):
        image_pad = torch.ones(3,image.shape[1]+width*2,image.shape[2]+width*2)*(colors[i,:,None,None]/255.0)
        image_pad[:,width:-width,width:-width] = image
        images_pad.append(image_pad)
    images_pad = torch.stack(images_pad,dim=0)
    return images_pad

@torch.no_grad()
def vis_cameras(opt,vis,step,poses=[],colors=["blue","magenta"],plot_dist=True, save_path=None):
    win_name = "{}/{}".format(opt.group,opt.name)
    data = []
    # set up plots
    centers = []
    for pose,color in zip(poses,colors):
        pose = pose.detach().cpu()
        vertices,faces,wireframe = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
        center = vertices[:,-1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode="markers",
            marker=dict(color=color,size=3),
        ))
        # colored camera mesh
        vertices_merged,faces_merged = merge_meshes(vertices,faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name,step),),
    ))

    # Create the figure data
    fig_data = dict(
        data=data,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            scene=dict(
                aspectmode='cube',  # This ensures equal scaling
                xaxis=dict(
                    showticklabels=False,  # Hide tick labels
                    title='X'
                ),
                yaxis=dict(
                    showticklabels=False,  # Hide tick labels
                    title='Y'
                ),
                zaxis=dict(
                    showticklabels=False,  # Hide tick labels
                    title='Z'
                ),
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
    )

    # Save to file if path is provided
    if save_path is not None:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        fig = go.Figure(data=data, layout=fig_data['layout'])
        
        # Set higher resolution (width and height in pixels)
        pio.write_image(fig, save_path, 
                       width=1920,    # Full HD width
                       height=1080,   # Full HD height
                       scale=2)       # Increase quality by scaling

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged
def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged
def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged

def plot_save_poses(opt,fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    setup_3D_plot(ax1,elev=-90,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    setup_3D_plot(ax2,elev=0,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    ax1.set_title("forward-facing view",pad=0)
    ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def plot_save_poses_blender(opt,fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title("epoch {}".format(ep),pad=0)
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    if ep==0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

def create_interactive_pose_animation(poses_over_time, color="magenta", save_path=None, duration=100, depth=1.0):
    """
    Creates an interactive visualization of camera poses with full camera meshes.
    Only shows the evolution of poses over time without ground truth comparison.
    
    Args:
        poses_over_time: tensor of shape [T, N, 3, 4] for T timesteps and N cameras
        color: color for the camera visualization
        save_path: path to save the HTML file
        duration: duration of each frame in milliseconds
        depth: depth of the camera frustum
    """
    # Convert to numpy for processing
    poses_over_time = poses_over_time.detach().cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Process each timestep
    frames = []
    for t, poses in enumerate(poses_over_time):
        frame_data = []
        
        # Process poses
        vertices, faces, wireframe = get_camera_mesh(torch.from_numpy(poses), depth=depth)
        center = vertices[:,-1]
        
        # Camera centers
        frame_data.append(go.Scatter3d(
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode='markers',
            marker=dict(color=color, size=3 * 0.1),
            name='Camera Centers'
        ))
        
        # Camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)
        frame_data.append(go.Mesh3d(
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
            name='Camera Mesh'
        ))
        
        # Camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        frame_data.append(go.Scatter3d(
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode='lines',
            line=dict(color=color),
            opacity=0.3 * 2,
            name='Camera Wireframe'
        ))
        
        # Create frame
        frames.append(go.Frame(
            data=frame_data,
            name=f'step_{t}'
        ))
    
    # Add first frame to figure
    fig.add_traces(frames[0].data)
    
    # Update layout with slider
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1),  # Force equal scaling for all axes
            xaxis=dict(
                showticklabels=False,  # Hide tick labels
                title=''
            ),
            yaxis=dict(
                showticklabels=False,  # Hide tick labels
                title=''
            ),
            zaxis=dict(
                showticklabels=False,  # Hide tick labels
                title=''
            )
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [[f'step_{k}' for k in range(len(frames))],
                            {'frame': {'duration': duration, 'redraw': True},
                             'mode': 'immediate',
                             'fromcurrent': True,
                             'transition': {'duration': 0}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None],
                            {'frame': {'duration': 0, 'redraw': False},
                             'mode': 'immediate',
                             'fromcurrent': True,
                             'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons'
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Optimization Step: '},
            'steps': [
                {
                    'args': [[f'step_{k}'], {'frame': {'duration': 0, 'redraw': True}}],
                    'label': str(k),
                    'method': 'animate'
                } for k in range(len(frames))
            ]
        }]
    )
    
    # Add frames to figure
    fig.frames = frames
    
    # Save to HTML file if path provided
    if save_path:
        fig.write_html(save_path)
    
    return fig

def create_interactive_camera_animation(poses_over_time, colors=["blue","magenta"], gt_poses=None, save_path=None, duration=100, depth=1.0):
    """
    Creates an interactive visualization of camera poses with full camera meshes.
    """
    # align poses
    gt_poses = gt_poses.detach().cpu().numpy()
    poses_over_time = poses_over_time.detach().cpu().numpy()
    for i in range(poses_over_time.shape[0]):
        poses_over_time[i] = align_poses(poses_over_time[i], gt_poses)
    poses_over_time = torch.from_numpy(poses_over_time).float()[:,:,:3,:]
    gt_poses = torch.from_numpy(gt_poses).float()[:,:3,:]

    selected_cameras = None
    #selected_cameras = [3, 210]
    if selected_cameras is not None:
        poses_over_time = poses_over_time[:, selected_cameras]
        gt_poses = gt_poses[selected_cameras]

    # Create figure
    fig = go.Figure()
    
    # Process each timestep
    frames = []
    for t, poses in enumerate(poses_over_time):
        #if t == 0:
            #colors = ["blue", "magenta"]
        #else:
            #colors = ["magenta", "blue"]
        frame_data = []
        centers = []
        
        # Process ground truth poses (blue)
        if gt_poses is not None:
            pose = gt_poses
            vertices, faces, wireframe = get_camera_mesh(pose, depth=depth)
            center = vertices[:,-1]
            centers.append(center)
            
            # Camera centers
            frame_data.append(go.Scatter3d(
                x=[float(n) for n in center[:,0]],
                y=[float(n) for n in center[:,1]],
                z=[float(n) for n in center[:,2]],
                mode='markers',
                marker=dict(color=colors[0], size=3 * 0.1),
                name='GT Centers'
            ))
            
            # Camera mesh
            vertices_merged, faces_merged = merge_meshes(vertices, faces)
            #frame_data.append(go.Mesh3d(
                #x=[float(n) for n in vertices_merged[:,0]],
                #y=[float(n) for n in vertices_merged[:,1]],
                #z=[float(n) for n in vertices_merged[:,2]],
                #i=[int(n) for n in faces_merged[:,0]],
                #j=[int(n) for n in faces_merged[:,1]],
                #k=[int(n) for n in faces_merged[:,2]],
                #flatshading=True,
                #color=colors[0],
                #opacity=0.05,
                #name='GT Mesh'
            #))
            for idx in range(len(vertices_merged) // 5):  # Divide by 5 since each camera has 5 vertices
                    frame_data.append(go.Mesh3d(
                        x=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,0]],
                        y=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,1]],
                        z=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,2]],
                        i=[0, 0, 0, 1, 2, 3],  # Fixed face indices for each camera
                        j=[1, 2, 1, 2, 3, 0],
                        k=[2, 3, 4, 4, 4, 4],
                        flatshading=True,
                        color=colors[0],
                        opacity=0.05,
                        name=f'GT Mesh_{idx}'
                    ))
            
            # Camera wireframe
            wireframe_merged = merge_wireframes(wireframe)
            frame_data.append(go.Scatter3d(
                x=wireframe_merged[0],
                y=wireframe_merged[1],
                z=wireframe_merged[2],
                mode='lines',
                line=dict(color='blue'),
                opacity=0.3 * 2,
                name='GT Wireframe'
            ))
        
        # Process estimated poses (red)
        pose = poses
        vertices, faces, wireframe = get_camera_mesh(pose, depth=depth)
        center = vertices[:,-1]
        centers.append(center)
        
        # Camera centers
        frame_data.append(go.Scatter3d(
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode='markers',
            marker=dict(color=colors[1], size=3*0.1),
            name='Est Centers'
        ))
        
        # Camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)
        #frame_data.append(go.Mesh3d(
            #x=[float(n) for n in vertices_merged[:,0]],
            #y=[float(n) for n in vertices_merged[:,1]],
            #z=[float(n) for n in vertices_merged[:,2]],
            #i=[int(n) for n in faces_merged[:,0]],
            #j=[int(n) for n in faces_merged[:,1]],
            #k=[int(n) for n in faces_merged[:,2]],
            #flatshading=True,
            #color=colors[1],
            #opacity=0.05,
            #name='Est Mesh'
        #))
        for idx in range(len(vertices_merged) // 5):  # Divide by 5 since each camera has 5 vertices
                frame_data.append(go.Mesh3d(
                    x=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,0]],
                    y=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,1]],
                    z=[float(n) for n in vertices_merged[idx*5:(idx+1)*5,2]],
                    i=[0, 0, 0, 1, 2, 3],  # Fixed face indices for each camera
                    j=[1, 2, 1, 2, 3, 0],
                    k=[2, 3, 4, 4, 4, 4],
                    flatshading=True,
                    color='blue',
                    opacity=0.05,
                    name=f'Est Mesh_{idx}'
                ))
        
        # Camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        frame_data.append(go.Scatter3d(
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode='lines',
            line=dict(color=colors[1]),
            #line=dict(color=colors[1], width=20),
            opacity=0.3 * 2,
            name='Est Wireframe'
        ))
        
        # Add distance lines between corresponding cameras
        if gt_poses is not None:
            center_merged = merge_centers(centers[:2])
            frame_data.append(go.Scatter3d(
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode='lines',
                line=dict(color='red', width=4),
                name='Distances'
            ))
        
        # Create frame
        frames.append(go.Frame(
            data=frame_data,
            name=f'step_{t}'
        ))
    
    # Add first frame to figure
    fig.add_traces(frames[0].data)
    
    # Update layout with slider
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1),  # Force equal scaling for all axes
            xaxis=dict(
                showticklabels=False,  # Hide tick labels
                title='',
                #showgrid=False,  # Hide grid
                #zeroline=False,  # Hide zero line
                #showline=False,  # Hide axis line
                #backgroundcolor="white"
            ),
            yaxis=dict(
                showticklabels=False,  # Hide tick labels
                title='',
                #showgrid=False,  # Hide grid
                #zeroline=False,  # Hide zero line
                #showline=False,  # Hide axis line
                #backgroundcolor="white"
            ),
            zaxis=dict(
                showticklabels=False,  # Hide tick labels
                title='',
                #showgrid=False,  # Hide grid
                #zeroline=False,  # Hide zero line
                #showline=False,  # Hide axis line
                #backgroundcolor="white"
            ),
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [[f'step_{k}' for k in range(len(frames))],
                            {'frame': {'duration': duration, 'redraw': True},
                             'mode': 'immediate',
                             'fromcurrent': True,
                             'transition': {'duration': 0}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None],
                            {'frame': {'duration': 0, 'redraw': False},
                             'mode': 'immediate',
                             'fromcurrent': True,
                             'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons'
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Optimization Step: '},
            'steps': [
                {
                    'args': [[f'step_{k}'], {'frame': {'duration': 0, 'redraw': True}}],
                    'label': str(k),
                    'method': 'animate'
                } for k in range(len(frames))
            ]
        }]
    )
    
    # Add frames to figure
    fig.frames = frames

    
    # Save to HTML file if path provided
    if save_path:
        fig.write_html(save_path)
    
    return fig






def align_poses(pose_a, pose_b):
    # Calculate alignment parameters using umeyama
    r, t, c = geometry.umeyama_alignment(pose_a[:,:3,3].T, pose_b[:,:3,3].T, with_scale=True)
            
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3,:3] = c * r  # Apply rotation and scale
    transform[:3,3] = t  # Add translation

    rotation = r
    aligned_a = np.einsum('ij,bjk->bik', transform, pose_a)
    aligned_a[:,:3,:3] = np.einsum('ij,bjk->bik', rotation, pose_a[:,:3,:3])

    return aligned_a

# Example usage:
"""
# Assuming you have:
poses_over_time = [...] # shape [T, N, 3, 4] for T timesteps and N cameras
gt_poses = [...] # shape [N, 3, 4]

# Create visualization
fig = create_interactive_camera_animation(
    poses_over_time=est_poses,  # [T, B, 3, 4]
    gt_poses=gt_poses,  # [B, 3, 4]
    save_path='camera_optimization.html',
    duration=0.05,
    depth=0.5,
)

"""