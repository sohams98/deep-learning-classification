#!/bin/sh

echo "Starting server by Gunicorn"
gunicorn -k gevent -w 1 --threads 2 -b 0.0.0.0:80 wsgi:_app

exec "$@"
