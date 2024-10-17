import gradio as gr
import requests

# Define the API URL
API_URL = "http://fastapi-app:8000"

# Example inputs for the Gradio interface
EXAMPLES = [
    ["Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"],
    ["Forest fire near La Ronge Sask. Canada"],
    ["All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"],
    ['13,000 people receive #wildfires evacuation orders in California'],
    ['Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school'],
    ['#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires'],
    ['#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas'],
    ["I'm on top of the hill and I can see a fire in the woods..."],
    ["There's an emergency evacuation happening now in the building across the street"],
    ["I'm afraid that the tornado is coming to our area..."]
]

# Function to list available models from the API
def list_models():
    response = requests.get(f"{API_URL}/models_list/")
    if response.status_code == 200:
        models = response.json()["models"]
        model_names = [f"{model['model_name']}_v{model['version']}.pkl" for model in models]
        return model_names
    else:
        return []

# Function to make a prediction using the selected model and input text
def make_prediction(model_name, raw_text):
    _, version = model_name.rsplit("_v", 1)
    version = version.rsplit(".", 1)[0]
    flag = model_name[:2]
    payload = {
        "model_name": model_name,
        "version": version,
        "raw_text": raw_text
    }
    if flag == "DL":
        # TODO: Implement the DL prediction
        pass
    else:
        response = requests.post(f"{API_URL}/predict_ml", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            if prediction == 0:
                return "Not a Disaster"
            else:
                return "Real Disaster"
        else:
            return "Error making prediction"

# Create the Gradio interface
def update_model_choices():
    return gr.Dropdown(choices=list_models())

with gr.Blocks() as demo:
    # Step 1: Create the dropdown without choices
    model_dropdown = gr.Dropdown(label="Select Model", choices=list_models())
    # Button to trigger the update of the dropdown choices
    # update_button = gr.Button("Update Models List")
    
    input_text = gr.Textbox(label="Input Text")
    predict_button = gr.Button("Make Prediction")
    
    output_text = gr.Textbox(label="Prediction", interactive=False)

    # Attach the event handler to the update button
    # update_button.click(fn=update_model_choices, inputs=[], outputs=model_dropdown)

    # Define the click event for the prediction button
    predict_button.click(make_prediction, inputs=[model_dropdown, input_text], outputs=output_text)

    # Add example inputs to the interface
    gr.Examples(examples=EXAMPLES, inputs=input_text)

# Launch the Gradio app
if __name__ == "__main__":
    # Specify the server name and port
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
