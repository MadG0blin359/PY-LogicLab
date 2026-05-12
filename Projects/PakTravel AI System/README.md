# 🚌 PakTravel AI System

> A comprehensive AI-driven travel management system built for Pakistan's intercity transport network, integrating classical AI techniques and modern machine learning into a single cohesive platform.

---

## 📌 Overview

PakTravel AI System is a multi-paradigm artificial intelligence project that tackles real-world transport logistics problems across Pakistan. It combines five distinct AI methodologies — from graph-based search and logical reasoning to neural networks and unsupervised learning — to deliver a complete smart travel solution.

The system covers:
- 🗺️ Optimal route planning across 15 major Pakistani cities
- ⚖️ AI-powered legal travel advisor
- 🕒 Automated bus and driver scheduling
- ⏱️ Bus delay prediction using a neural network
- 👥 Traveller segmentation via clustering

---

## 👨‍💻 Authors

| Name | Role |
|------|------|
| Shawaiz Shahzad | Developer |
| Waleed Afzal | Developer |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **NumPy** | ANN & backpropagation from scratch |
| **NetworkX** | Graph modeling & visualization of the travel network |
| **Scikit-learn** | K-Means Clustering |
| **Jupyter Notebook** | Development & interactive execution environment |

---

## 🧠 AI Modules & Techniques

The system is divided into **5 independent modules**, each powered by a different AI paradigm:

### 1. 🗺️ Route Planning — Search Algorithms
Finds the most efficient travel path between 15 major cities in Pakistan using a weighted graph where cities are nodes and roads are edges with distance weights.

- **Algorithms Used:** Uniform Cost Search (UCS), A\* Search, Bidirectional Search
- **Finding:** A\* search is more computationally efficient than UCS, reaching the same optimal path while exploring fewer nodes.

---

### 2. ⚖️ AI Legal Advisor — Propositional Logic
A knowledge-based system that guides travellers on Pakistan travel regulations, legal requirements, and restrictions.

- **Technique:** Knowledge Base + Entailment (Propositional Logic)
- Answers queries about travel rules by reasoning over a structured knowledge base.

---

### 3. 🕒 Bus Scheduling — Constraint Satisfaction Problem (CSP)
Automates the assignment of drivers and buses to routes while respecting availability, timing constraints, and operational limits.

- **Technique:** Backtracking Search + Constraint Propagation
- Treats bus timings and driver availability as CSP variables with strict domain constraints.

---

### 4. ⏱️ Delay Prediction — Artificial Neural Network (ANN)
Estimates bus arrival delays by analyzing external factors such as weather conditions, traffic density, and route distance.

- **Architecture:** Feed-forward MLP — 3 input neurons → 5 hidden neurons → 1 output neuron
- **Implementation:** Built from scratch using NumPy (no high-level ML framework)
- **Performance:** Mean Squared Error (MSE) of **12.4**

---

### 5. 👥 Traveller Segmentation — K-Means Clustering
Segments 200 traveller records based on spending behavior and trip frequency to enable targeted loyalty rewards.

- **Technique:** K-Means Clustering (via Scikit-learn)
- **Dataset:** 200 dummy records with `Spending Score` and `Trips per Month` features
- **Result:** 4 distinct traveller clusters identified — **Loyal**, **Regular**, **Casual**, and **Occasional**

---

## 📁 Project Structure

```
PakTravel AI System/
│
├── PakTravel_AI_App.ipynb               # Main Jupyter Notebook (all modules)
├── PakTravel AI System - Project Report.pdf   # Full project report
├── PakTravel AI System - Project Report.docx  # Editable report document
├── PakTravel AI System Presentation.pdf       # Project presentation slides
└── README.md                                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

```bash
pip install numpy networkx scikit-learn jupyter
```

### Running the System

1. Clone or download the project folder.
2. Open a terminal in the project directory.
3. Launch Jupyter Notebook:

```bash
jupyter notebook PakTravel_AI_App.ipynb
```

4. Run the cells **sequentially** from top to bottom to initialize all modules.
5. Interact with each module by providing inputs (e.g., source/destination cities for routing, queries for the legal advisor).

---

## 📊 Results Summary

| Module | Technique | Key Result |
|--------|-----------|------------|
| Route Planning | A\* / UCS / Bidirectional | A\* is most efficient — fewer nodes explored |
| Legal Advisor | Propositional Logic | Correctly resolves travel regulation queries |
| Bus Scheduling | CSP + Backtracking | Valid driver-bus assignments generated |
| Delay Prediction | ANN (from scratch) | MSE of **12.4** |
| Traveller Segmentation | K-Means | 4 clusters: Loyal, Regular, Casual, Occasional |

---

## 🔭 Future Work

- 🗺️ **Google Maps API Integration** — Real-time navigation with live traffic data
- 📱 **Mobile Application** — Frontend app for end-user accessibility
- 🤖 **Reinforcement Learning** — Dynamic, demand-based ticket pricing

---

## 📄 License

This project was developed as part of an academic course. All rights belong to the respective authors.

---

*Built with ❤️ in Pakistan 🇵🇰*
