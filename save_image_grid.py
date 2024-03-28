import os
from PIL import Image

def save_image_grid(folder_path, output_path, grid_size=(4, 2), image_size=(512, 512)):
    # 폴더 내의 모든 이미지 파일 목록을 가져옵니다.
    images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg'))]
    
    # 지정된 그리드 크기에 맞게 첫 8개의 이미지만 사용합니다.
    images = images[:grid_size[0] * grid_size[1]]
    
    # 각 이미지를 로드하고 크기를 조정합니다.
    images = [Image.open(image).resize(image_size) for image in images]
    
    # 빈 캔버스를 생성합니다. 캔버스의 크기는 그리드 크기와 각 이미지 크기에 따라 달라집니다.
    canvas_width = image_size[0] * grid_size[0]
    canvas_height = image_size[1] * grid_size[1]
    canvas = Image.new('RGB', (canvas_width, canvas_height))
    
    # 각 이미지를 캔버스에 붙입니다.
    for index, image in enumerate(images):
        row = index // grid_size[0]
        col = index % grid_size[0]
        canvas.paste(image, (col * image_size[0], row * image_size[1]))
    
    # 결과 이미지를 파일로 저장합니다.
    canvas.save(output_path)

# 함수를 호출하여 이미지 그리드를 생성하고 저장합니다.
# 'your_image_folder_path'를 이미지가 저장된 폴더 경로로, 'output_image_path.jpg'를 원하는 출력 파일 이름으로 바꾸세요.
save_image_grid('./examples/hand_generation_text_with_various_prompt', './examples/grid_hand.png')
