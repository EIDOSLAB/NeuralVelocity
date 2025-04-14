#!/bin/sh

docker build -t eidos-service.di.unito.it/dalmasso/neve:base . -f Dockerfile
docker push eidos-service.di.unito.it/dalmasso/neve:base

docker build -t eidos-service.di.unito.it/dalmasso/neve:python . -f Dockerfile.python
docker push eidos-service.di.unito.it/dalmasso/neve:python

docker build -t eidos-service.di.unito.it/dalmasso/neve:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/dalmasso/neve:sweep