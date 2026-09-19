import requests
import sys
import os

# Add local path for config
sys.path.append(os.path.dirname(__file__))
from shared import config

def update_endpoint_image(endpoint_id, new_image_tag):
    print(f"Updating RunPod Endpoint {endpoint_id} to use image: {new_image_tag}...")
    
    # Official RunPod GraphQL endpoint
    url = "https://api.runpod.io/graphql"
    
    # GraphQL Mutation to update the endpoint image via saveEndpoint
    # Note: saveEndpoint is used to both create and update.
    mutation = """
    mutation {
      saveEndpoint(input: {
        id: "%s",
        imageName: "%s"
      }) {
        id
        imageName
      }
    }
    """ % (endpoint_id, new_image_tag)
    
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json={'query': mutation}, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if 'errors' in result:
            print(f"Error updating endpoint: {result['errors']}")
            return False
            
        updated_data = result.get('data', {}).get('saveEndpoint', {})
        if not updated_data:
            print("Failed to receive confirmation from RunPod API.")
            return False

        print(f"Successfully triggered update for {updated_data.get('id')}")
        print(f"New Image: {updated_data.get('imageName')}")
        return True
    except Exception as e:
        print(f"Failed to call RunPod API: {e}")
        return False

if __name__ == "__main__":
    # The new image name we just pushed to Docker Hub
    new_image = "winterfa/yolov11-runpod:latest"
    success = update_endpoint_image(config.RUNPOD_ENDPOINT_ID, new_image)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
