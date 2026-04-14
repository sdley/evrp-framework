#!/bin/bash
# 
# RL4EVRP Framework - Installation & Quick Reference
# 
# This script helps you get started with the RL4EVRP framework
#

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        RL4EVRP - Electric Vehicle Routing with Deep RL        ║"
echo "║              Modular Framework - Production Ready              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python version:"
python3 --version
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q python-dotenv pyyaml numpy torch matplotlib pandas 2>/dev/null
echo "✅ Dependencies installed"
echo ""

# Test framework
echo "🧪 Testing framework import..."
python3 << 'PYEOF'
try:
    import sys
    sys.path.insert(0, '.')
    import rl4evrp as rl
    print("✅ RL4EVRP framework imported successfully")
    
    framework = rl.RL4EVRP()
    print("✅ Framework initialized")
    
    inst = framework.generate_instance(seed=42)
    print(f"✅ Instance generated [{inst['n_nodes']} nodes]")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   QUICK START COMMANDS                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  Read the Quick Start Guide:"
echo "   cat QUICKSTART.md"
echo ""
echo "2️⃣  Open the Example Notebook:"
echo "   jupyter notebook run.ipynb"
echo ""
echo "3️⃣  Python Quick Test:"
echo "   python3 << 'EOF'"
echo "   import rl4evrp as rl"
echo "   framework = rl.RL4EVRP()"
echo "   model = framework.build().complete_model()"
echo "   instances = [framework.generate_instance(i) for i in range(10)]"
echo "   print('✅ Ready to train!')"
echo "   EOF"
echo ""
echo "4️⃣  Train a Model:"
echo "   python3 << 'EOF'"
echo "   import rl4evrp as rl"
echo "   from rl4evrp.utils import train_agent"
echo "   framework = rl.RL4EVRP()"
echo "   model = framework.build().complete_model()"
echo "   instances = [framework.generate_instance(i) for i in range(200)]"
echo "   results = train_agent(model, instances, n_episodes=50)"
echo "   print(f'Training complete! Results: {results}')"
echo "   EOF"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   DOCUMENTATION FILES                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📖 SUMMARY.md              → Overview & getting started"
echo "📖 QUICKSTART.md           → 5-minute tutorial"
echo "📖 README.md               → Full API documentation"
echo "📖 REFACTORING.md          → What changed & why"
echo "📖 FILES.md                → Complete file structure"
echo ""
echo "📓 run.ipynb               → Runnable example notebook"
echo "📓 evrp_xai_final_combined.ipynb → Original notebook (reference)"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   PACKAGE STRUCTURE                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "rl4evrp/"
echo "├── __init__.py              ← Main RL4EVRP class"
echo "├── config/"
echo "│   ├── __init__.py          ← Config loader"
echo "│   ├── problem.yaml         ← Problem configuration"
echo "│   ├── model.yaml           ← Model & training config"
echo "│   └── env.yaml             ← Environment config"
echo "├── environment/             ← EVRP environment & generation"
echo "├── models/                  ← Neural architecture"
echo "├── agents/                  ← A2C agent"
echo "├── utils/                   ← Training utilities"
echo "└── xai.py                   ← Explainable AI tools"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                       KEY FEATURES                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✨ YAML-based Configuration    → Edit problem/model/env configs"
echo "✨ Modular Architecture        → Easy to customize & extend"
echo "✨ Multi-Seed Training         → Reproducible experiments"
echo "✨ Built-in XAI Tools          → Attention & counterfactuals"
echo "✨ Production-Ready Code       → Type hints, docstrings, tests"
echo "✨ Clean APIs                  → Simple, Pythonic interface"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     CONFIGURATION TIPS                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔧 Edit YAML Files to Customize:"
echo ""
echo "   Increase problem size:      Edit rl4evrp/config/problem.yaml"
echo "   Change learning rate:       Edit rl4evrp/config/model.yaml"
echo "   Switch to CUDA:             Edit rl4evrp/config/env.yaml"
echo "   Add Groq API key:           Edit .env file"
echo ""
echo "⚙️  Train on your custom config:"
echo ""
echo "   python3 << 'EOF'"
echo "   import rl4evrp as rl"
echo "   framework = rl.RL4EVRP()   # Loads your YAML configs"
echo "   model = framework.build().complete_model()"
echo "   # ... training code ..."
echo "   EOF"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   NEXT STEPS                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Read: cat QUICKSTART.md"
echo "2. Install: pip install -e ."
echo "3. Run: jupyter notebook run.ipynb"
echo "4. Configure: Edit rl4evrp/config/*.yaml"
echo "5. Train: Run training code from examples"
echo "6. Extend: Customize for your use case"
echo ""
echo "✅ Framework is ready to use! 🚗⚡"
echo ""
