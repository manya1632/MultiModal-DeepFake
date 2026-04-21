/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    PYTHON_BACKEND_URL: process.env.PYTHON_BACKEND_URL || 'http://localhost:8000',
  },
}
module.exports = nextConfig
