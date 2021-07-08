# importing libraries
import paho.mqtt.client as paho
import os
import io
import h5py
import socket
import ssl
from tensorflow import keras
import random
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from time import sleep
from random import uniform
from sklearn.metrics import accuracy_score
import sys

def initialize_topic(num_rpis):
    topic = "home/ModelRP"
    topic_list = []
    for i in range (num_rpis):
        topic_list.append(topic + str(i+1))
    return topic_list


def fit_model(model):
    history = model.fit(X_train, y_train, batch_size=8, epochs=1, verbose=1, validation_split=0.2)
    model.save('SavedModel.h5')
    return model

def define_model():
    input_layer = tf.keras.layers.Flatten(input_shape=(X.shape[1],))
    dense_layer_1 = tf.keras.layers.Dense(15, activation = 'relu')
    dense_layer_2 = tf.keras.layers.Dense(10, activation = 'relu')
    output = tf.keras.layers.Dense(y.shape[1],activation='softmax')

    #model = Model(inputs=input_layer, outputs=output)
    model = tf.keras.models.Sequential([
     input_layer,
     dense_layer_1,
     dense_layer_2,
     output
    ])
    return model


def setup_data():
    cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety','output']
    cars = pd.read_csv(r'car_dataset.csv', names=cols, header=None)

    price = pd.get_dummies(cars.price, prefix='price')
    maint = pd.get_dummies(cars.maint, prefix='maint')
    doors = pd.get_dummies(cars.doors, prefix='doors')
    persons = pd.get_dummies(cars.persons, prefix='persons')
    lug_capacity = pd.get_dummies(cars.lug_capacity, prefix='lug_capacity')
    safety = pd.get_dummies(cars.safety, prefix='safety')
    labels = pd.get_dummies(cars.output, prefix='condition')

    return cars,price,maint,doors,persons,lug_capacity,safety,labels

def get_model(payload):
    f=io.BytesIO(payload)
    h=h5py.File(f, 'r')
    new_model=keras.models.load_model(h)
    f.close()
    return new_model

def model_testing(model):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(y_test, axis=1))
    print('global_acc: {:.3%} | global_loss: {}'.format( acc, loss))

    #score=model.evaluate(X_test, y_test, verbose=1)
    #print("Test Score:", score[0])
    #print("Test Accuracy:", score[1])


def on_connect(client, userdata, flags, rc):                # func for making connection
    global connflag
    print("Connected to AWS")
    connflag = True
    print("Connection returned result: " + str(rc))


def publish_model(model):
    model.save('SavedModel.h5')
    fo = open(filename, "rb")
    ByteArray = fo.read()
    print("msg sent: home/AWS")
    mqttc.publish("home/AWS", ByteArray, qos=1)

def global_vars():
     global clientKeys
     global upLimit
     global local_models_weight_list
     clientKeys = 0
     upLimit = 0
     local_models_weight_list = []


def scale_model_weights(weight, scalar): #Función para escalar los pesos de los modelos
    weight_final = []
    weight_num = len(weight)
    for i in range(weight_num):
        weight_final.append(scalar * weight[i]) #se multiplica cada peso por el escalar que 
                                                #define que porcentaje de los datos totales pertenecen a este cliente
    return weight_final

def average_model(weights_list): #Función para promediar los pesos de los modelos
    avg_grad = list()
    for tuples in zip(*weights_list):
        layer_mean = tf.math.reduce_sum(tuples, axis=0) #suma de elementos del tensor
        avg_grad.append(layer_mean) #se van añadiendo los pesos promediados
    
    return avg_grad



def on_message(client, userdata, msg):                      # Func for receiving msgs
    global upLimit
    global local_models_weight_list 
    print("Message received")
    print("topic: "+ msg.topic)
    RPINumber = int(msg.topic[-1])
    model = get_model(msg.payload)
    model_testing(model)
    scaled_weights = scale_model_weights(model.get_weights(),1/2)
    local_models_weight_list.append(scaled_weights)
    upLimit += RPINumber
    if upLimit == clientKeys:
        print("All models arrived check")
        upLimit = 0
        avg_weights = average_model(local_models_weight_list)
        model.set_weights(avg_weights)
        print("Testing global model")
        model_testing(model)
        publish_model(model)


def getMAC(interface='eth0'):
    # Return the MAC address of the specified interface
    try:
        str = open('/sys/class/net/%s/address' % interface).read()
    except:
        str = "00:00:00:00:00:00"
    return str[0:17]



def getEthName():
    # Get name of the Ethernet interface
    try:
        for root, dirs, files in os.walk('/sys/class/net'):
            for dir in dirs:
                if dir[:3] == 'enx' or dir[:3] == 'eth':
                    interface = dir
    except:
        interface = "None"
    return interface



if __name__ == "__main__":

    import logging
    logging.getLogger('tensorflow').disabled = True
    connflag = False
    filename = "SavedModel.h5"  # file to send
   
    global_vars()

    cars,price,maint,doors,persons,lug_capacity,safety,labels = setup_data()

    X = pd.concat([price, maint, doors, persons, lug_capacity, safety] , axis=1)

    y = labels.values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    from tensorflow.keras.layers import Input, Dense, Activation,Dropout
    from tensorflow.keras.models import Model

    print("Creating model")

    model = define_model()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


    # def on_log(client, userdata, level, buf):
#    print(msg.topic+" "+str(msg.payload))


    mqttc = paho.Client()                                       # mqttc object
    # assign on_connect func
    mqttc.on_connect = on_connect
    # assign on_message func
    mqttc.on_message = on_message
    #mqttc.on_log = on_log

    #### *Common for publ and subs* ####
    awshost = "a3f3tevlek9dbt-ats.iot.us-east-1.amazonaws.com"      # Endpoint
    awsport = 8883                                              # Port no.
    clientId = "PabloClient"                                     # Thing_Name
    thingName = "PabloClient"                                    # Thing_Name
    # Root_CA_Certificate_Name
    caPath = "/home/ubuntu/modelComm/root-ca.pem"
    # <Thing_Name>.cert.pem
    certPath = "/home/ubuntu/modelComm/certificate.pem.crt"
    # <Thing_Name>.private.key
    keyPath = "/home/ubuntu/modelComm/private.pem.key"

    mqttc.tls_set(caPath, certfile=certPath, keyfile=keyPath, cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)  # pass parameters

    # connect to aws server
    mqttc.connect(awshost, awsport, keepalive=60)

    ### *Till here* ###

    sleep(3)
    mqttc.loop_start()                                          # Start the loop
    ethName = getEthName()
    ethMAC = getMAC(ethName)
    macIdStr = ethMAC

    publish_model(model)
    topic_list = initialize_topic(sys.argv[0])

    for x in range(0,len(topic_list)):
        mqttc.subscribe(topic_list[x],1)
        clientKeys = clientKeys + x + 1


    while 1 == 1:
        if connflag == True:
            p=1

        else:
            print("waiting for connection...")

