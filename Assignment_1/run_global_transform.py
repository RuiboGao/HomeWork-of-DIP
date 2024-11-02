import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)

    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    # image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    
    if flip_horizontal:
        image=cv2.flip(image,1)

    h,w=image.shape[:2] # 获得图像的高度和宽度
    rad=np.pi*rotation/180 # 将输入的旋转角度由度转为弧度单位

    # 方便计算生成alpha和beta变量
    alpha=scale*np.cos(rad) 
    beta=scale*np.sin(rad)
    center = np.round(np.array([w/2,h/2])) # 确认图像的中心坐标
    cenx=center[0]
    ceny=center[1]

    # 计算中心旋转的仿射变换矩阵
    affine_matrix=np.array([[alpha,beta,(1-alpha)*cenx-beta*ceny+pad_size+translation_x],[-beta,alpha,beta*cenx+(1-alpha)*ceny+pad_size+translation_y]])
    
    # 计算仿射变换矩阵的前两列矩阵的逆
    M_inv=np.linalg.inv(affine_matrix[:,:2])

    # 对目标图像的像素遍历
    for i in range(image_new.shape[0]):
        for j in range(image_new.shape[1]):

            # 获得像素的坐标dst_pos
            dst_x,dst_y=j,i
            dst_pos=np.array([dst_x,dst_y]).reshape([2,1])

            # 通过仿射变换的逆变换，计算目标图像中的像素在原图像中的对应像素坐标src_pos
            src_pos=M_inv@(dst_pos-affine_matrix[:,-1].reshape([2,1]))
            ox,oy=src_pos[0],src_pos[1]

            # 通过双线性插值获得目标图像的像素近似
            if ox>=0 and ox<w-1 and oy>=0 and oy<h-1:
                low_ox = int(np.floor(ox)) # 向下取整
                low_oy = int(np.floor(oy)) # 向下取整
                high_ox = low_ox + 1 # 向上取整
                high_oy = low_oy + 1 # 向上取整
                # 双线性插值
                # p0        p1
                #      0
                # p2        p3
                # 针对图片来说
                pos = ox - low_ox, oy - low_oy # 获取相对位置

                # 获取四点的权重
                p0_area = (1 - pos[0]) * (1 - pos[1])
                p1_area = pos[0] * (1 - pos[1])
                p2_area = (1 - pos[0]) * pos[1]
                p3_area = pos[0] * pos[1]

                # 获取四点的坐标
                p0 = np.array([low_ox, low_oy])
                p1 = np.array([high_ox, low_oy])
                p2 = np.array([low_ox, high_oy])
                p3 = np.array([high_ox, high_oy])

                # 获取四点原图像中的像素值
                p0_value = image[p0[1], p0[0]]
                p1_value = image[p1[1], p1[0]]
                p2_value = image[p2[1], p2[0]]
                p3_value = image[p3[1], p3[0]]

                # 计算目标图像的像素值
                image_new[i,j] = p0_area * p0_value + p1_area * p1_value + p2_area * p2_value + p3_area * p3_value

    return image_new


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
