# DEPLOYMENT GUIDE - RUNPOD T4 SERVERLESS

Follow these steps to deploy the YOLOv11L inference server to RunPod.

## LOCAL PREPARATION
1. Ensure you have the RunPod SDK and requests installed:
   ```bash
   pip install -r requirements.txt
   ```

## DOCKER DEPLOYMENT
1. **Build the Docker Image**:
   Open a terminal in the project root and run:
   ```bash
   docker build -t yolov11-runpod .
   ```

2. **Tag the Image**:
   > [!IMPORTANT]
   > Docker repository names MUST be lowercase. Replace `yourusername` with your actual Docker Hub username in lowercase:
   ```bash
   docker tag yolov11-runpod yourusername/yolov11-runpod:latest
   ```

3. **Push to Docker Hub**:
   ```bash
   docker push yourusername/yolov11-runpod:latest
   ```

## RUNPOD CONFIGURATION
1. Log in to your RunPod Dashboard.
2. Navigate to **Serverless** -> **Endpoints**.
3. Create a new Endpoint or Edit an existing one:
   - **Endpoint Name**: `GPU Video Inference Optimizer`
   - **Docker Image**: `DOCKERHUB_USERNAME/yolov11-runpod:latest`
   - **GPU Type**: `NVIDIA T4` (16GB VRAM)
   - **Active Workers**: Set according to your concurrency needs.
4. Save and Deploy.
5. Note the **Endpoint ID** and ensure it matches the `RUNPOD_ENDPOINT_ID` in `shared/config.py`.

## VERIFICATION
1. Run the standalone test script to ensure the connection works:
   ```bash
   python test_endpoint.py
   ```
2. Once the test passes, you can launch the full pipeline:
   ```bash
   python main.py
   ```
