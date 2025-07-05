# This script initializes a MongoDB database with a user and a database.
# It waits for the MongoDB service to be ready before executing the commands.
wait_for_db=4 



# ------ Functions ------
check_mongo() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.adminCommand('ping')" --quiet
}

#Check if database exists
check_db() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.getMongo().getDBNames().indexOf('$MONGO_DB') >= 0" --quiet
}
#Check if user_collection exists
check_user_collection() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.getMongo().getDB('$MONGO_DB').getCollectionNames().indexOf('$MONGO_USER_COLLECTION') >= 0" --quiet
}

#Check if ratings_collection exists
check_ratings_collection() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.getMongo().getDB('$MONGO_DB').getCollectionNames().indexOf('$MONGO_RATINGS_COLLECTION') >= 0" --quiet
}

#Check if chat_collection exists
check_chat_collection() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.getMongo().getDB('$MONGO_DB').getCollectionNames().indexOf('$MONGO_CHAT_COLLECTION') >= 0" --quiet
}

#Create database
create_db() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS <<EOF > dev/null 2>&1
use $MONGO_DB
db.createCollection('$MONGO_USER_COLLECTION')
db.createCollection('$MONGO_RATINGS_COLLECTION')
db.createCollection('$MONGO_CHAT_COLLECTION')
EOF
}

#Create User 
create_user() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS <<EOF > dev/null 2>&1
use $MONGO_DB
db.createUser({
  user: '$MONGO_USER',
  pwd: '$MONGO_PASS',
  roles: [
    { role: 'readWrite', db: '$MONGO_DB' }
  ]
})
EOF
}

#Check if User exists
check_user() {
mongosh --port $MONGO_PORT --username $MONGO_USER --password $MONGO_PASS --eval "db.getMongo().getDB('$MONGO_DB').getUser('$MONGO_USER')" --quiet
}
# ------ Main Script ------
echo "Initializing MongoDB database..."
sleep $wait_for_db

if check_mongo | grep -q "ok"; then
    echo "MongoDB is running."
else
    echo "MongoDB is not running. Exiting."
    exit 1
fi

echo "Checking if database '$MONGO_DB' exists..."
if check_db; then
    echo "Database '$MONGO_DB' already exists."
else
    echo "Database '$MONGO_DB' does not exist. Creating database..."
    create_db 
    if ["$(check_db)" = "true"]; then
        echo "Database '$MONGO_DB' and collections created successfully."
    else
        echo "Failed to create database '$MONGO_DB'. Exiting."
        exit 1
    fi

echo "Checking if User '$MONGO_USER' exists..."
if check_user; then
    echo "User '$MONGO_USER' already exists."
else
    echo "User '$MONGO_USER' does not exist. Creating user..."
    create_user
    if [ "$(check_user)" ]; then
        echo "User '$MONGO_USER' created successfully."
    else
        echo "Failed to create user '$MONGO_USER'. Exiting."
        exit 1
    fi
fi
    