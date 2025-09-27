const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const cors = require("cors");

const app = express();
const PORT = 3001;

// ✅ Enable CORS for your frontend (development mode)
app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps or curl requests)
      if (!origin) return callback(null, true);
      
      // Allow any localhost or 127.0.0.1 origin for development
      if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
        return callback(null, true);
      }
      
      // For production, you'd want to be more restrictive
      return callback(null, true); // Allow all for development
    },
    methods: ["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
    credentials: true,
    optionsSuccessStatus: 200 // Some legacy browsers choke on 204
  })
);

// ✅ Handle preflight for all routes
app.use((req, res, next) => {
  if (req.method === 'OPTIONS') {
    cors()(req, res, next);
  } else {
    next();
  }
});

// ✅ Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// ✅ Multer storage config
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage: storage });

// ✅ Serve uploaded files statically
app.use("/uploads", express.static(uploadDir));

// ✅ POST /upload → upload file
app.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }
  res.json({
    message: "File uploaded successfully",
    filename: req.file.filename,
    path: req.file.path,
  });
});

// ✅ GET /files → list uploaded files
app.get("/files", (req, res) => {
  fs.readdir(uploadDir, (err, files) => {
    if (err) return res.status(500).json({ error: "Failed to read directory" });
    res.json({ files });
  });
});

// ✅ GET /download/:filename → download file
app.get("/download/:filename", (req, res) => {
  const filePath = path.join(uploadDir, req.params.filename);
  res.download(filePath, (err) => {
    if (err) {
      res.status(404).json({ error: "File not found" });
    }
  });
});

// ✅ Catch-all route (Express 5 safe)
app.get(/.*/, (req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// ✅ Start server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});