INSTANCE_NAME="instance-1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ir-assignment3-480114"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="sagihor" # without the @post.bgu.ac.il or @gmail.com part

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project $PROJECT_NAME
# gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
INSTANCE_IP=$(gcloud compute addresses describe $IP_NAME --region=$REGION --format="get(address)")

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=e2-micro \
  --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
  --metadata-from-file startup-script=startup_script_gcp.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output $INSTANCE_NAME --zone $ZONE

# After Ctrl+C, run steps 4-8 manually. 
# Depending on the way you ran this script, you may need to define again 
# the variables from the top (INSTANCE_NAME, etc.) in the console.
# Verify that the instance is running
gcloud compute instances list --filter="name=$INSTANCE_NAME" --format="table(name,status,zone,EXTERNAL_IP)"

# 4. Secure copy your app to the VM (assume search_frontend.py is available in the current directory)
gcloud compute scp ./search_frontend.py \
  ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME}:/home/${GOOGLE_ACCOUNT_NAME} \
  --zone ${ZONE}

# 5. SSH to your VM and start the app
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE

# 6. Verify the enviroment is all set (run in the VM)
# ~/venv/bin/python - << 'PY'
# import flask, werkzeug, numpy, pandas
# print("flask:", flask.__version__)
# print("werkzeug:", werkzeug.__version__)
# print("numpy:", numpy.__version__)
# print("pandas:", pandas.__version__)
# PY

# 7. Run the server
nohup ~/venv/bin/python ~/search_frontend.py > ~/frontend.log 2>&1 &

# 8. Start querying
curl "http://127.0.0.1:8080/search?query=hello"

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q $INSTANCE_NAME
# make sure there are no lingering instances
gcloud compute instances list
delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
delete external addresses
gcloud compute addresses delete -q $IP_NAME --region $REGION