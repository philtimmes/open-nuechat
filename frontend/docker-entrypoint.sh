#!/bin/sh
set -e

# Substitute environment variables in nginx config
envsubst '${FRONTEND_PORT} ${BACKEND_PORT}' < /etc/nginx/templates/nginx.conf.template > /etc/nginx/conf.d/default.conf

# Start nginx
exec nginx -g 'daemon off;'
