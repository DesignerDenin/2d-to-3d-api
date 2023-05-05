from flask import Flask, request, send_file
from flask_cors import CORS
from generate import Generate
from PIL import Image
from rembg import remove
import shutil
import io
import os

app = Flask(__name__)
CORS(app)

@app.route('/remove-bg', methods=['POST'])
def edit_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Removing Background
    image = Image.open(request.files['image'])
    trans_output = remove(image)

    # Cropping transparent parts
    bbox = trans_output.getbbox()
    if bbox:
        trans_output = trans_output.crop(bbox)
    else:
        trans_output = trans_output

    # Adding white background
    background = Image.new('RGBA', (200, 200), (255, 255, 255, 255))
    trans_output.thumbnail(background.size)
    width, height = trans_output.size
    trans_output = trans_output.resize((int(width * 0.8), int(height * 0.8)))

    # Pasting image to white background
    x = (background.width - trans_output.width) // 2
    y = (background.height - trans_output.height) // 2
    background.paste(trans_output, (x, y), trans_output)

    # Convert the edited image to bytes
    output = io.BytesIO()
    background.save(output, format='PNG')
    output.seek(0)

    return send_file(output, mimetype='image/png')

@app.route("/generate", methods=['POST'])
def generate():
    image_name = "image.png"
    image_path = os.path.join(os.getcwd(), "others", image_name)
    generation_path = os.path.join(os.getcwd(), "others/generation")

    if os.path.exists(generation_path):
        shutil.rmtree(generation_path)
    
    if os.path.exists(image_path):
        os.remove(image_path)

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)

    Generate("configs/demo.yaml", True)
    model_path = os.path.join(generation_path, "meshes/na", image_name + ".glb")

    return send_file(model_path, mimetype='model/gltf-binary')

@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome <h1>"

if __name__ == "__main__":
    app.run()
