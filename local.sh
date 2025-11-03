port="8888"

# Define environment variables that will be used by MyST
# We'll use the values of these variables in our Jupyter server as well.
export JUPYTER_BASE_URL="http://localhost:${port}"
export JUPYTER_TOKEN="my-jupyter-token"

# Start the Jupyter server re-using the variables above
jupyter server --IdentityProvider.token="${JUPYTER_TOKEN}" --ServerApp.port="${port}" &
myst start --execute