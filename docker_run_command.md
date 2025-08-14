---

## **4️⃣ Run Instructions**

### **Build the Docker image**

```bash
docker-compose build
```

### **Run the Flask app**

```bash
docker-compose up sentiment-app
```

Then open: **[http://localhost:5000](http://localhost:5000)**

### **Run tests inside Docker**

```bash
docker-compose run --rm sentiment-tests
```

This will:

* Run **all 10 pytest cases**
* Generate **`report.html`** with detailed logs
* Show **Expected vs Actual results** in console output

---