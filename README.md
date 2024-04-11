# MLWorkshop & Student AI
## Project: Student AI (RAG Pipeline)

--- 

## Project: Image-to-Audio with Gemini and ElevenLabs

This project demonstrates how to capture an image using your webcam, convert it to text using the Gemini API, and generate audio from the text using ElevenLabs.

### **Prerequisites:**

- **Google Colab account:** Sign up for a free account at https://colab.research.google.com.

- **Gemini API key:** Obtain a key from https://aistudio.google.com/app/apikey.

- **ElevenLabs API key:** Get a key from https://www.elevenlabs.io.

### **Installation:**

1. Open a Colab notebook.

2. Run the following commands in a code cell:

```python
!pip install -q elevenlabs -U
!pip install -q google-generativeai
```
### Code Explanation:

1. #### Import Libraries:

```python
import elevenlabs
import PIL
import google.generativeai as genai
import time
from elevenlabs import generate
from google.colab import userdata
from IPython.display import Audio, display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
```

2. #### API Key Configuration:

```python
# Api keys
genai.configure(api_key=userdata.get('gemini'))  # Gemini
elevenlabs.set_api_key(userdata.get('elevenlabs'))  # ElevenLabs
```

3. #### Model Setup:

```python
generation_config = {
  "temperature": 0.5,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 512,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

# Gemini model
model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config,
                              safety_settings=safety_settings)
```

4. #### Capture Image Function:

```python
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve)=>{
        setTimeout(()=>{return resolve("Done")}, 5000)
      })

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
```

5. #### Analyze Image Function:

```python
def analyze_image(image, script):
  input_text = "{\n\"role\": \"system\",\n\"content\": \"\"\"You are Sir David Attenborough. Narrate the picture of the human as if it is a nature documentary.Make it snarky and funny. Don't repeat yourself. Make it short about 300 words. If I do anything remotely interesting, make a big deal about it!\"\"\",}" + script + "{\n\"role\": \"user\",\n\"content\": [{\"type\": \"text\", \"text\": \"Describe this image.\"},\n]\n}"
  global response
  response = model.generate_content([input_text, image])
  response.resolve()
  try:
    return response.candidates[0].content.parts[0].text
  except:
    print("An exception occurred")
    return ""
```

6. #### Play audio using 11Labs:
```python
def play_audio(text):
  audio = generate(
    text=text,
    voice="Daniel",
    model="eleven_multilingual_v1"
  )
  display(Audio(audio, autoplay=True))
  time.sleep(max(2,(len(audio)/16429.0) - 15))
```
7. #### Connecting all in single function:

```python
def camera2audio():
  script = ""
  while True:
    # path to your image
    image_path = "photo.jpg"

    # Taking Image from camera
    filename = take_photo(filename = image_path)
    print("📸 Say cheese! Saving frame.")
    display(Image(filename))

    # opening image using PIL
    img = PIL.Image.open(image_path)

    # analyze posture
    print("👀 David is watching...")
    analysis = analyze_image(img, script=script)

    # print(analysis)
    if(analysis!=""):
      print("🎙️ David says:")
      play_audio(analysis)

      script = script + f"\n{analysis}"
```

8. #### Run:

```python
camera2audio()
```
