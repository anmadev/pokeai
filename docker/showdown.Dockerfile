FROM node:22-alpine

WORKDIR /app

# Pokemon Showdown server
# Pinned to commit d4ba2e66ae10295e4fbaaa35985d5760d687ef81 (2026-04-19)
# Commit: "Champions OU: Ban Starmie-Mega, Lucario-Mega and Palafin"
COPY . .

# Install dependencies
RUN npm install

EXPOSE 8000

# --no-security disables authentication requirements
# essential for local bots — they don't have PS accounts
CMD ["node", "pokemon-showdown", "start", "--no-security"]
