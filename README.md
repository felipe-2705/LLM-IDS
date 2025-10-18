<h1 align="center">📌 Welcome to XXXXX-FS! 📌</h1>

<h4 align="left">
✔️ XXXXX-FS is a command-line tool that applies the GRASP metaheuristic with a priority queue, changing random process of feature decision to a complex LLM model, to perform feature selection in Intrusion Detection Systems (IDS). It was designed to work with enriched datasets, focusing on performance and reproducibility.
</h4>

<h2>📁 Repository Structure</h2>
<pre><code>.
├── data/                     # Folder for training and test datasets
├── results/                  # Examples of logs and generated plots from execution
├── utils/                    # Folder with python library for data loading, preprocessing, evalutation, priority queue, logging, graph ploting and Arg parser.
├── Dockerfile                # Optional Docker image for containerized execution
├── main.py                   # Main script to run the XXXXX-FS algorithm
├── FeatureSelectorLLM.py     # Custom feature selector based on LLM Requests
├── requirements.txt          # List of required Python packages
├── README.md                 # This documentation file
</code></pre>

<h2>📋 Index</h2>
<ol>
  <li>Test Environment</li>
  <li>Requirements</li>
  <li>Development Environment</li>
  <li>Usage Example</li>
  <li><a href="#portuguese">🇧🇷 Versão em português!</a></li>
</ol>

<h3>🖱️ Test Environment</h3>
<table border="1">
<tr><th>Configuration</th><th>Machine</th></tr>
<tr><td>Operating System</td><td>Windows 10</td></tr>
<tr><td>Processor</td><td>Intel Core i7-7700 3.60GHz</td></tr>
<tr><td>RAM</td><td>16 GB</td></tr>
<tr><td>Python Version</td><td>3.10.0</td></tr>
</table>

<h3>⚙️ Development Environment</h3>
<table border="1">
<tr><th>Tool</th><th>Version</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code</td></tr>
<tr><td>Terminal</td><td>PowerShell or CMD</td></tr>
</table>

<h3>📝 Requirements</h3>
<p>This Python project uses the following libraries:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
  <li>groq ≥ 0.2.0</li>
</ul>

<h3>🚀 How to Run</h3>

<h4>▶️ Option 1: Run Locally (Recommended for Development)</h4>
<p>To get started with this project locally, follow these steps:</p>
<ol>
  <li>
    <strong>Clone this repository and enter the project folder:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>
  <li>
    <strong>Create a virtual environment (recommended):</strong>
    <pre><code>python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate  # on Unix/Mac</code></pre>
  </li>
  <li>
    <strong>Install the dependencies:</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>
    <strong>Run the tool with the desired configuration:</strong>
    <pre><code>python main.py -a rf -rcl 40 -is 10 -pq 10 -lc 5 -cb 20 -lb 20</code></pre>
  </li>
</ol>

<p><strong>Other usage examples:</strong></p>
<pre><code>
python main.py -a nb -rcl 40 -is 10 -pq 10 -lc 5 -cb 20 -lb 20
or
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 5 -cb 40 -lb 10
</code></pre>

<h4>🐳 Option 2: Run with Docker (No Python Installation Required)</h4>
<p>This option is useful for fast execution without installing dependencies:</p>
<ol>
  <li>
    <strong>Clone this repository and enter the project folder:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>
  <li>
    <strong>Build the Docker image:</strong>
    <pre><code>docker build -t main .</code></pre>
  </li>
  <li>
    <strong>Run the container:</strong>
    <pre><code>docker run --rm main -a nb -rcl 40 -is 10 -pq 10 -lc 5 -cb 20 -lb 20</code></pre>
  </li>
</ol>

<p>ℹ️ The <code>Dockerfile</code> is included in the root of this repository.</p>

<p><strong>Parameters (with all aliases):</strong></p>
<ul>
  <li><code>-a</code>, <code>--algorithm</code>, <code>--alg</code>: Classifier (<code>nb</code>, <code>dt</code>, <code>knn</code>, <code>rf</code>, <code>svm</code>, <code>linear_svc</code>, <code>sgd</code>, <code>xgboost</code>)</li>
  <li><code>-rcl</code>, <code>--rcl_size</code>, <code>--rcl</code>: Restricted Candidate List size</li>
  <li><code>-is</code>, <code>--init_sol</code>, <code>--initial_solution</code>: Number of features in the initial solution</li>
  <li><code>-pq</code>, <code>--pq_size</code>, <code>--priority-queue</code>: Size of the priority queue</li>
  <li><code>-cb</code>, <code>--const</code>, <code>--constructive_batch</code>: Number of constructive solutions requested in the first batch</li>
  <li><code>-lb</code>, <code>--local_batch</code>, <code>--localbatch</code>: Number of solutions in the local search phase batch.</li>
  <li><code>-lc</code>, <code>--ls</code>, <code>--local_iterations</code>: Number of local search iterations per solution</li>
  <li><code>-d</code>, <code>--debug</code>: Enable debug mode.</li>
</ul>

<p>⚠️ Ensure that the datasets <code>hibrid_dataset_GOOSE_train.csv</code> and <code>hibrid_dataset_GOOSE_test.csv</code> are inside the <code>data/</code> folder.</p>

<hr>