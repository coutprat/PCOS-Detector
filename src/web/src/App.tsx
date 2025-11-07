import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import MockModeBanner from "./components/MockModeBanner";
import Home from "./pages/Home";
import ImageOnly from "./pages/ImageOnly";
import EHROnly from "./pages/EHROnly";
import Hybrid from "./pages/Hybrid";
import "./styles/theme.css";

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <MockModeBanner />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/image" element={<ImageOnly />} />
            <Route path="/ehr" element={<EHROnly />} />
            <Route path="/hybrid" element={<Hybrid />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
