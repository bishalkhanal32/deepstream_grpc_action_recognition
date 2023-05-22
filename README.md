# STEPS

- Clone this repo inside Documents folder
- Clone models from minIO and put it inside the main folder
- Run copyModel.sh script to copy models and test videos to their respective folders
- Run main_script.sh script to create two docker containers- one for DeepStream and another for Classification service. It will run the classification service automatically. But for DeepStream App you need to go inside the container.


## Running DeepStream App
- docker exec -it container_name bash
- Go inside /opt/nvidia/deepstream/deepstream-6.2/sources/apps/Deepstream-app/ in DeepStream container and run setup_deepsort.sh script to put DeepSORT models in respective place.
- Go inside one of the project folder (e.g. deepstream\_yolo\_vitpose\_grpc) 
- Run build_grpc.sh script.
- Run run.sh script.


## Cloning models from minIO
- Setup MinIO client

	1. $ wget https://dl.min.io/client/mc/release/linux-amd64/mc
	2. $ chmod +x mc && sudo mv mc /usr/local/bin
	3. $ mc alias set ec2 http://192.168.5.244:9000 eldercare eldercareAdmin

- cd to the project's main folder

- Download models from MinIO
	1. $ mc cp --recursive ec2/ec2-model ./
