#!/usr/bin/env python3

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import logging
import paho.mqtt.client as paho
import os
import io
import h5py
import math
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


def global_vars():
    '''
    Initialize all the global variables
    '''

    global rondacomm
    global bytearrays_list
    global received_chunks
    global model
    global ACK
    global all_clients
    global trained

    trained = False
    all_clients = False
    ACK = False
    bytearrays_list = [bytearray()] * 200
    rondacomm = 1
    received_chunks = [int()] * 100


def clean_vars():
    '''
    Clean the variables at each comm round
    '''

    global bytearrays_list
    global received_chunks
    global rondacomm
    global all_clients
    global trained
    received_chunks = [int()] * 100
    bytearrays_list = [bytearray()] * 200
    rondacomm = rondacomm + 1
    all_clients = False
    trained = False


def fit_model(model):
    '''
    Function for model training
    '''

    model.fit(x_train, y_train, batch_size=8,
              epochs=int(sys.argv[4]), verbose=0, validation_split=0.2)
    model.save("./data/SavedModel" + nclient + ".h5")
    return model

def setup_data():
    '''
    Extract the data from the dataset
    '''
    
    data = np.load('./data/mnist' + str(nclient) + '.npz')
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # Preprocessing
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add one domention to make 3D images
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, x_test, y_test, y_train


def get_model(payload):
    '''
    Construct the model from the bytearray
    '''


    f = io.BytesIO(payload)
    try:
        h = h5py.File(f, 'r')
        new_model = keras.models.load_model(h)
        f.close()
        return new_model
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
        print("Error ", str(e))
        return False


def model_testing(model):
    '''
    Get the accuracy of the global model
    '''

    eval_loss, eval_acc = model.evaluate(x_test,  y_test, verbose=1)
    print('Eval accuracy percentage: {:.2f}'.format(eval_acc * 100))

    """     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(x_test)
        loss = cce(y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(y_test, axis=1))
        print('global_acc: {:.3%} | global_loss: {}'.format(acc, loss)) """
    """## Evaluation"""

  


def get_lengths(model):
    '''
    Get the lengths of the model in order to publish every chunk
    :return sliceLength: the length of each chunk that is going to be publish, except the last one
    :return finalSlice: the length of the last chunk
    :return ByteArray: the whole model converted to bytearray
    :return arrayLength: the length of the bytearray
    :return slicesNumber: the number of chunks that the model is divided into
    '''

    model.save(filename)
    fo = open(filename, "rb")
    ByteArray = fo.read()
    arrayLength = len(ByteArray)
    slicesNumber = math.ceil(arrayLength/128000)
    sliceLength = math.ceil(arrayLength/slicesNumber)
    if (arrayLength % slicesNumber) == 0:  # Check if each piece is whole
        finalSlice = sliceLength
    else:
        resto = (arrayLength % slicesNumber)
        #If there is remainder, the last slice must be smaller
        finalSlice = sliceLength - slicesNumber + resto
    return sliceLength, finalSlice, ByteArray, arrayLength, slicesNumber


def on_connect(client, userdata, flags, rc):          
    '''
    Function for making connection 
    '''

    global connflag
    print("Connected to AWS")
    connflag = True
    mqttc.subscribe("home/AWSF", qos=1)
    mqttc.subscribe("home/AWSC" + str(nsuscriber), qos=1)
    print("home/AWSC" + str(nsuscriber))


def publish_model(model):
    '''
    Publish the model to a topic
    '''

    sliceLength, finalSlice, ByteArray, arrayLength, slicesNumber = get_lengths(
        model)

    mqttc.publish("home/ModelRP" + str(nclient),  bytearray(str(slicesNumber), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(
        str(rondacomm), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(str(slicesNumber) + "*", "UTF-8") + ByteArray[arrayLength - finalSlice:arrayLength], qos=1)
    sleep(int(nclient))
    for i in range(slicesNumber-1, 0, -1):
        mqttc.publish("home/ModelRP" + str(nclient), bytearray(str(i), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(
            str(rondacomm), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(str(slicesNumber) + "*", "UTF-8") + ByteArray[sliceLength*(i-1):sliceLength*i], qos=1)
        sleep(int(nclient))
    print("All chunks were sent client" + str(nclient))


def message_values(payload, topic):
    '''
    Get the differents params needed from the message
    :return chunkID: the ID from the chunk, from 1 to the total number of chunks.
    :return rComm: the round of communication in which the message was sent
    :return NumberSlices: the number of chunks the model is divided into
    :return limite+1: to determine which part of the payload is meta information
    '''
    limite = 0
    for i in range(0, 20):
        if (chr(payload[i]).isdigit() or chr(payload[i]) == '/'):
            limite += 1
        else:
            break
    tmp = str(payload[0:limite].decode("utf-8"))
    tmp = tmp.split("/")
    chunkID = int(tmp[0])
    rComm = int(tmp[1])
    NumberSlices = int(tmp[2])
    return chunkID, rComm, NumberSlices, (limite+1)


def check_if_complete(NumberSlices):
    '''
    Function to check if every chunk of a client has arrived
    '''

    global bytearrays_list
    tmp = True

    for i in range(1, NumberSlices + 1):
        if received_chunks[i] == 0:
            tmp = False

    return tmp


def on_message(client, userdata, msg):                   
    '''
    Function for receiving msgs
    '''
    
    global bytearrays_list
    global rondacomm
    global model
    global ACK
    global all_clients
    global trained

    if msg.topic[-1] == "F" :
        all_clients = True
    
    if trained == True and all_clients == True:
        sleep(2*int(nclient))
        publish_model(model)                    
        clean_vars()
    elif msg.topic[-1] != "F" :
        chunkID, rComm, NumberSlices, limite = message_values(
            msg.payload, msg.topic)
        rondacomm = rComm

        if received_chunks[chunkID] == 0:  # That chunk hasn't arrived yet
            ACK = True
            received_chunks[chunkID] = 1
            # add each piece to the total
            bytearrays_list[chunkID] = msg.payload[limite:]
            tmp = check_if_complete(NumberSlices)
            # If true, that means it is the last chunk needed from that client
            if tmp == True:
                aux_bytearray = bytearray()
                print("mensaje completo cliente", nclient)
                for i in range(NumberSlices, 0, -1):   #rebuilding the model
                    aux_bytearray[0:0] = bytearrays_list[i]
                model_o = get_model(aux_bytearray)
                if model_o != False:   #in the case that the model is rebuilt correctly
                    model = model_o
                    model = fit_model(model)
                    trained = True
                else: #the previous model is published
                    model = fit_model(model)
                    trained = True                        


def getMAC(interface='eth0'):
    '''
    Return the MAC address of the specified interface
    '''
    
    try:
        str = open('/sys/class/net/%s/address' % interface).read()
    except:
        str = "00:00:00:00:00:00"
    return str[0:17]


def getEthName():
    '''
    Get name of the Ethernet interface
    '''    
    try:
        for root, dirs, files in os.walk('/sys/class/net'):
            for dir in dirs:
                if dir[:3] == 'enx' or dir[:3] == 'eth':
                    interface = dir
    except:
        interface = "None"
    return interface

if __name__ == "__main__":

    global_vars()
    nclient = sys.argv[2]
    print("ncliente: ", nclient)
    nrpi = int(sys.argv[3])
    nsuscriber = int(sys.argv[5])
    connflag = False
    global rondacomm
    global ACK
    filename = "./data/SavedModel" + nclient + ".h5"  # file to send
    logging.getLogger('tensorflow').disabled = True
    x_train, x_test, y_test, y_train = setup_data()
  
    # mqqtc object
    mqttc = paho.Client(client_id=str(nclient))
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message

    #### *Common for publ and subs* ####
    awshost = "a3f3tevlek9dbt-ats.iot.us-east-1.amazonaws.com"      # Endpoint
    awsport = 8883                                              # Port no.
    clientId = "PabloClient"                                     # Thing_Name
    thingName = "PabloClient"                                    # Thing_Name
    # Root_CA_Certificate_Name
    caPath = "/home/pi/IoTCloud/root-ca.pem"
    # <Thing_Name>.cert.pem
    certPath = "/home/pi/IoTCloud/certificate.pem.crt"
    # <Thing_Name>.private.key
    keyPath = "/home/pi/IoTCloud/private.pem.key"

    mqttc.tls_set(caPath, certfile=certPath, keyfile=keyPath, cert_reqs=ssl.CERT_REQUIRED,
                  tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)  # pass parameters

    # connect to aws server
    mqttc.connect(awshost, awsport, keepalive=60)

    ethName = getEthName()
    ethMAC = getMAC(ethName)
    macIdStr = ethMAC
    mqttc.loop_start()                                          # Start the loop

    while 1 == 1:
        if connflag == True:
            if (ACK == True):
                ACK = False
        else:
            p = 2
