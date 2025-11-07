# PCOS Detection Lab

A production-ready, medical-themed React web application for research-grade PCOS (Polycystic Ovary Syndrome) detection using advanced AI models.

## Features

- **Three Detection Modes:**

  - **Image-Only**: Upload ultrasound images for computer vision-based detection
  - **EHR-Only**: Fill out medical history and lab values for risk assessment
  - **Hybrid**: Combine image and EHR data for comprehensive AI-driven diagnosis

- **Advanced Features:**

  - Grad-CAM visualization for interpretability
  - Calibrated probability scores
  - Adjustable prediction thresholds
  - Real-time API health monitoring
  - Mock mode for offline development

- **Design:**
  - Premium medical theme with glassmorphism effects
  - Responsive, mobile-first layout
  - Accessible UI with ARIA support
  - Smooth animations with Framer Motion

## Tech Stack

- **Frontend:** React 18 + TypeScript + Vite
- **Routing:** React Router
- **Styling:** TailwindCSS with custom medical theme
- **Forms:** React Hook Form + Zod validation
- **Animations:** Framer Motion
- **API:** Axios with automatic fallback
- **Icons:** Lucide React

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:3000`

### Environment Variables

Create a `.env` file in the root directory:

```bash
# API Configuration
VITE_API_BASE=http://127.0.0.1:8000  # Your FastAPI backend URL

# Optional: Force mock mode
VITE_USE_MOCK=1  # Set to "1" to always use mock mode
```

### Mock vs Real API

The app automatically falls back to mock mode if the backend is unavailable. This allows you to develop and test the UI without a running backend.

**Mock Mode:**

- Deterministic hash-based predictions
- Stable test data
- Fully functional UI
- Shown with a yellow banner when active

**Real API Mode:**

- Connects to FastAPI backend at `VITE_API_BASE`
- Live model predictions
- Health status shown in navbar

## Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── Alert.tsx
│   ├── Card.tsx
│   ├── Footer.tsx
│   ├── GradCamViewer.tsx
│   ├── MetricBadge.tsx
│   ├── MockModeBanner.tsx
│   ├── Navbar.tsx
│   ├── Spinner.tsx
│   ├── ThresholdSlider.tsx
│   └── UploadBox.tsx
├── lib/             # Utilities and API
│   ├── api.ts       # API client with fallback
│   ├── ehrSchema.ts # EHR data schema (Zod)
│   ├── env.ts       # Environment variables
│   ├── mock.ts      # Mock API implementation
│   └── utils.ts     # Helper functions
├── pages/           # Page components
│   ├── Home.tsx
│   ├── ImageOnly.tsx
│   ├── EHROnly.tsx
│   └── Hybrid.tsx
├── styles/
│   └── theme.css    # Global styles & Tailwind
├── App.tsx          # Main app component
└── main.tsx         # Entry point
```

## API Endpoints

The app expects the following FastAPI endpoints:

- `GET /health` - Health check and device info
- `POST /predict` - Image-only prediction (multipart)
- `POST /predict_ehr` - EHR-only prediction (JSON)
- `POST /predict_hybrid` - Hybrid prediction (multipart + JSON)
- `POST /calibrate` - Calibrate probability score
- `POST /gradcam` - Generate Grad-CAM overlay

See `src/lib/api.ts` for detailed request/response types.

## Building for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

```bash
npm run preview
```

Preview the production build locally.

## Development Guidelines

- TypeScript strict mode enabled
- ESLint configured for React best practices
- All components typed with TypeScript interfaces
- Accessible by default (ARIA labels, keyboard navigation)
- Responsive design tested on mobile and desktop

## Disclaimer

**Research Prototype. Not for Clinical Use.**

This tool is developed for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment.

## License

MIT
