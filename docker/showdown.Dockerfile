FROM node:22-alpine

WORKDIR /app

# Copy the cloned server into the image
COPY . .

# Install dependencies
RUN npm install

EXPOSE 8000

# --no-security disables authentication requirements
# essential for local bots — they don't have PS accounts
CMD ["node", "pokemon-showdown", "start", "--no-security"]
