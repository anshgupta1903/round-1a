# 📦 round-1a
This project processes input data and writes results to an output directory using a Dockerized application.

# 🐳 Build Docker Image
To build the Docker image for the project (using the linux/amd64 platform):


docker build --platform linux/amd64 -t round-1a .
# ▶️ Run the Docker Container
To run the container:

" docker run --rm `
  -v "${PWD}/input:/app/input:ro" `
  -v "${PWD}/output/repoidentifier:/app/output" `
  --network none round-1a "

# ✅ Explanation of flags:

--rm: Remove the container after it exits.

-v "${PWD}/input:/app/input:ro": Mounts the local input folder as read-only to /app/input in the container.

-v "${PWD}/output/repoidentifier:/app/output": Mounts the local output/repoidentifier folder to /app/output in the container.

--network none: Disables networking for extra isolation.

round-1b: Name of the Docker image to run.

📂 Folder Structure

round-1a/
├── input/                  # Place your input files here
├── output/
│   └── repoidentifier/     # Results will be saved here
├── Dockerfile
└── README.md
📝 Notes
Make sure you have Docker installed and running.

Replace round-1b with your actual image name if it differs.

Always check that the input and output/repoidentifier folders exist before running.