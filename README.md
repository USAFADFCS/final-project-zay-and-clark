[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nwy6MBDZ)
# FAIR-LLM Installation Guide

## 🚀 Quick Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository (for demos)

```bash
git clone git@github.com:USAFA-AI-Center/fair_llm_demos.git
cd fair-llm-demos
```

### Step 2: Install All Dependencies

Simply install everything needed using the requirements file:

```bash
pip install -r requirements.txt
```

This will install:
- `fair-llm>=0.1` - The core FAIR-LLM package
- `python-dotenv` - For environment variable management
- `rich` - For beautiful terminal output
- `anthropic` - For Anthropic Claude integration
- `faiss-cpu` - For vector search capabilities
- `seaborn` - For data visualization
- `pytest` - For testing

### Step 3: Set Up API Keys

Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Or create a new one
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
```

Or export them as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### Step 4: Verify Installation

Run the verification script:

```bash
python verify_setup.py
```

You should see a colorful output showing all components are properly installed!

## 🎯 Running the Demos

Once installed, try the demo scripts:

### Essay Autograder Demo
```bash
# Basic grading
python demos/demo_committee_of_agents_essay_autograder.py \
  --essays essay_autograder_files/essays_to_grade/ \
  --rubric essay_autograder_files/grading_rubric.txt \
  --output essay_autograder_files/graded_essays/

# With RAG fact-checking
python demos/demo_committee_of_agents_essay_autograder.py \
  --essays essay_autograder_files/essays_to_grade/ \
  --rubric essay_autograder_files/grading_rubric.txt \
  --output essay_autograder_files/graded_essays/ \
  --materials essay_autograder_files/course_materials/
```

### Code Autograder Demo
```bash
# Static analysis only (safer)
python demos/demo_committee_of_agents_coding_autograder.py \
  --submissions coding_autograder_files/submissions/ \
  --rubric coding_autograder_files/rubric.txt \
  --output coding_autograder_files/reports/ \
  --no-run

# With test execution (requires sandbox)
python demos/demo_committee_of_agents_coding_autograder.py \
  --submissions coding_autograder_files/submissions/ \
  --tests coding_autograder_files/tests/test_calculator.py \
  --rubric coding_autograder_files/rubric.txt \
  --output coding_autograder_files/reports/
```

## 📦 Upgrading

To upgrade to the latest versions:

```bash
# Upgrade all packages
pip install --upgrade -r requirements.txt

# Or just upgrade fair-llm
pip install --upgrade fair-llm
```

## 🐛 Troubleshooting

### Missing Dependencies
If you get import errors, ensure all requirements are installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### API Key Issues
The demos will create sample files if they don't exist, but ensure your API keys are set:
```python
python -c "import os; print('OpenAI Key:', 'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set')"
```

### Virtual Environment Issues
Always use a virtual environment to avoid conflicts:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 What's Included

After installation, you'll have:
- ✅ The complete FAIR-LLM framework
- ✅ Multi-agent orchestration capabilities
- ✅ Document processing tools
- ✅ Vector search with FAISS
- ✅ Beautiful terminal output with Rich
- ✅ Complete demo applications

## 🎉 Next Steps

1. Run `python verify_setup.py` to confirm everything is working
2. Explore the `demos/` folder for examples
3. Set up and run some demos
4. Start building your own multi-agent demo files!

## 👥 Contributors
Developed by the USAFA AI Center team:

Ryan R (rrabinow@uccs.edu)
Austin W (austin.w@ardentinc.com)
Eli G (elijah.g@ardentinc.com)
Chad M (Chad.Mello@afacademy.af.edu)
