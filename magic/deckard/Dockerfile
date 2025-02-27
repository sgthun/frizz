# Stage 1: Build React app
FROM node:16 AS build
WORKDIR /app
# Copy package files and install dependencies
COPY /deckard/package.json ./
RUN npm install
# Copy the rest of your source code and build the React app
COPY ./deckard/* .
RUN npm run build

CMD ["npm", "start"]