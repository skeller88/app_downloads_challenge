# Google Cloud Environment
Download `gcloud`.

Create auth key file and download to local machine. Copy to `goodwater_challenge/.gcs/credentials.json`.
The `.gcs` folder is gitignored.

```bash
export KEY_FILE=[your-key-file]
gcloud auth activate-service-account --key-file=$KEY_FILE
gcloud auth configure-docker

export PROJECT_ID=[your-project-id]
export HOSTNAME=us.gcr.io
export BASE_IMAGE_NAME=$HOSTNAME/$PROJECT_ID
```

# Model training
## Prototype locally with Jupyter notebook
```
# Prepare a hashed password:
# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password
export JUPYTER_PASSWORD_SHA=[your-hashed-password-from-above-step]
export FILEDIR=machine_learning/jupyter_tensorflow_notebook
export IMAGE_NAME=$BASE_IMAGE_NAME/jupyter_tensorflow_notebook
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile --build-arg jupyter_password_sha_build_arg=$JUPYTER_PASSWORD_SHA .
docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work $IMAGE_NAME
docker push $IMAGE_NAME
```

## Create GCP instance from Google image family
```
# scopes needed are pub/sub, service control, service management, container registry,
# stackdriver logging/trace/monitoring, storage
# Full names: --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/logging.admin,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/source.read_only \
export DISK_NAME=disk
export DISK_MOUNT_PATH=/mnt/disk
export FILEDIR=machine_learning/jupyter_tensorflow_notebook
export IMAGE_PROJECT=deeplearning-platform-release
export IMAGE_FAMILY=common-cu100
gcloud compute instances create jupyter-tensorflow-notebook \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-8 \
        --boot-disk-size=50GB \
        --metadata=enable-oslogin=TRUE,install-nvidia-driver=True \
        --metadata-from-file=startup-script=$FILEDIR/startup_script.sh \
        --disk=name=$DISK_NAME,auto-delete=no,mode=rw,device-name=$DISK_NAME \
        --tags http-server

# Upload model notebook from local to instance
gcloud compute scp ~/Documents/goodwater_challenge/machine_learning/model.ipynb jupyter-tensorflow-notebook:$DISK_MOUNT_PATH

# Navigate to the instance IP address and login to the notebook. There should be a jupyter-tensorflow-notebook/model.ipynb
# notebook file in the home directory. Open the notebook and run it to train the models.

# If you make changes to the model.ipynb notebook while running it on the instance, download the notebook from the
# instance to your local machine to sync changes.
gcloud compute scp jupyter-tensorflow-notebook:$DISK_MOUNT_PATH/model.ipynb ~/Documents/goodwater_challenge/machine_learning
```

# Model deployment
```
export FILEDIR=machine_learning/model_server
export IMAGE_NAME=$BASE_IMAGE_NAME/model_server
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile  .
docker run -it --rm --env-file $FILEDIR/env.list -p 8889:8889 $IMAGE_NAME
docker push $IMAGE_NAME

gcloud compute instances create-with-container model-server \
        --zone=us-west1-b \
        --can-ip-forward \
        --container-image=$IMAGE_NAME \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-1 \
        --boot-disk-size=1GB \
        --metadata enable-oslogin=TRUE \
        --tags http-server

## make request
python machine_learning/model_server/make_test_request.py
```
