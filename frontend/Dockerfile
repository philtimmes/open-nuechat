# Frontend Dockerfile
# Build stage - Using Node.js 22 LTS for security
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Remove any old Tailwind/PostCSS config files (Tailwind v4 uses Vite plugin)
RUN rm -f postcss.config.js postcss.config.cjs postcss.config.mjs \
    tailwind.config.js tailwind.config.cjs tailwind.config.mjs tailwind.config.ts

# Build the application
# Note: VITE_API_URL and VITE_WS_URL should NOT be set here
# The frontend auto-detects URLs from window.location at runtime
RUN npm run build

# Production stage
FROM nginx:alpine

# Default ports (can be overridden via environment)
ENV FRONTEND_PORT=80
ENV BACKEND_PORT=8000

# Copy nginx template and entrypoint
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

ENTRYPOINT ["/docker-entrypoint.sh"]
