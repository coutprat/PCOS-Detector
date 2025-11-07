export default function Footer() {
  return (
    <footer className="bg-white/5 border-t border-white/10 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center text-white/60 space-y-4">
          <p className="text-sm">PCOS Detection Lab - Research Prototype</p>
          <p className="text-xs italic">
            This tool is for research and educational purposes only. Not for
            clinical use.
          </p>
          <div className="flex justify-center gap-6 text-xs">
            <a href="#" className="hover:text-white transition-colors">
              About
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Privacy
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Terms
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
