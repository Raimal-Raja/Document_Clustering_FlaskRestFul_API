from flask import Flask, render_template, jsonify, request
import sqlite3
import os

app = Flask(__name__)
DB_PATH = "clustering.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        DROP TABLE IF EXISTS DocumentClusters;
        DROP TABLE IF EXISTS DocumentTags;
        DROP TABLE IF EXISTS Documents;
        DROP TABLE IF EXISTS Clusters;

        CREATE TABLE Documents (
            DocumentID INT PRIMARY KEY,
            Title VARCHAR(255),
            Content TEXT
        );

        CREATE TABLE Clusters (
            ClusterID INT PRIMARY KEY,
            ClusterName VARCHAR(100)
        );

        CREATE TABLE DocumentClusters (
            DocumentID INT,
            ClusterID INT,
            PRIMARY KEY (DocumentID, ClusterID),
            FOREIGN KEY (DocumentID) REFERENCES Documents(DocumentID),
            FOREIGN KEY (ClusterID) REFERENCES Clusters(ClusterID)
        );

        CREATE TABLE DocumentTags (
            DocumentID INT,
            Tag VARCHAR(50),
            PRIMARY KEY (DocumentID, Tag),
            FOREIGN KEY (DocumentID) REFERENCES Documents(DocumentID)
        );
    """)

    documents = [
        (1,  'D1',  'Artificial Intelligence is transforming modern technology.'),
        (2,  'D2',  'Machine learning algorithms improve data analysis.'),
        (3,  'D3',  'Cloud computing helps store large amounts of data.'),
        (4,  'D4',  'Cybersecurity protects computer systems from attacks.'),
        (5,  'D5',  'Cricket is the most popular sport in Pakistan.'),
        (6,  'D6',  'Football is played worldwide.'),
        (7,  'D7',  'The Olympics include many international sports.'),
        (8,  'D8',  'Players train hard to improve performance.'),
        (9,  'D9',  'Universities provide higher education.'),
        (10, 'D10', 'Students attend lectures and complete assignments.'),
        (11, 'D11', 'Online learning platforms are growing rapidly.'),
        (12, 'D12', 'Teachers help students understand complex subjects.'),
        (13, 'D13', 'Regular exercise improves health.'),
        (14, 'D14', 'Doctors recommend balanced diets.'),
        (15, 'D15', 'Hospitals provide medical treatment.'),
        (16, 'D16', 'Vaccines prevent many diseases.'),
        (17, 'D17', 'Climate change affects global temperatures.'),
        (18, 'D18', 'Trees help reduce pollution.'),
        (19, 'D19', 'Recycling helps protect the environment.'),
        (20, 'D20', 'Renewable energy reduces carbon emissions.'),
    ]
    c.executemany("INSERT INTO Documents VALUES (?,?,?)", documents)

    clusters = [
        (1, 'Technology'),
        (2, 'Sports'),
        (3, 'Education'),
        (4, 'Health'),
        (5, 'Environment'),
    ]
    c.executemany("INSERT INTO Clusters VALUES (?,?)", clusters)

    doc_clusters = [
        (1,1),(2,1),(3,1),(4,1),
        (5,2),(6,2),(7,2),(8,2),
        (9,3),(10,3),(11,3),(12,3),
        (13,4),(14,4),(15,4),(16,4),
        (17,5),(18,5),(19,5),(20,5),
    ]
    c.executemany("INSERT INTO DocumentClusters VALUES (?,?)", doc_clusters)

    tags = [
        (1,'AI'),(2,'ML'),(3,'Cloud'),(4,'Cybersecurity'),
        (5,'Cricket'),(6,'Football'),(7,'Olympics'),(8,'Sports'),
        (9,'University'),(10,'Students'),(11,'OnlineLearning'),(12,'Teaching'),
        (13,'Exercise'),(14,'Diet'),(15,'Hospital'),(16,'Vaccine'),
        (17,'Climate'),(18,'Trees'),(19,'Recycling'),(20,'Energy'),
    ]
    c.executemany("INSERT INTO DocumentTags VALUES (?,?)", tags)

    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/clusters')
def api_clusters():
    conn = get_db()
    rows = conn.execute("SELECT ClusterID, ClusterName FROM Clusters").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/cluster_counts')
def api_cluster_counts():
    conn = get_db()
    rows = conn.execute("""
        SELECT c.ClusterName, COUNT(dc.DocumentID) as DocCount
        FROM Clusters c
        JOIN DocumentClusters dc ON c.ClusterID = dc.ClusterID
        GROUP BY c.ClusterName
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/documents_by_cluster')
def api_documents_by_cluster():
    conn = get_db()
    rows = conn.execute("""
        SELECT c.ClusterName, c.ClusterID, d.DocumentID, d.Title, d.Content
        FROM Documents d
        JOIN DocumentClusters dc ON d.DocumentID = dc.DocumentID
        JOIN Clusters c ON dc.ClusterID = c.ClusterID
        ORDER BY c.ClusterName
    """).fetchall()
    conn.close()
    result = {}
    for r in rows:
        name = r['ClusterName']
        if name not in result:
            result[name] = {'ClusterID': r['ClusterID'], 'docs': []}
        result[name]['docs'].append({'id': r['Title'], 'content': r['Content']})
    return jsonify(result)

@app.route('/api/tags')
def api_tags():
    conn = get_db()
    rows = conn.execute("""
        SELECT t.Tag, d.Title
        FROM Documents d
        JOIN DocumentTags t ON d.DocumentID = t.DocumentID
        ORDER BY t.Tag
    """).fetchall()
    conn.close()
    result = {}
    for r in rows:
        tag = r['Tag']
        if tag not in result:
            result[tag] = []
        result[tag].append(r['Title'])
    return jsonify(result)

@app.route('/api/stats')
def api_stats():
    conn = get_db()
    total_docs     = conn.execute("SELECT COUNT(*) as n FROM Documents").fetchone()['n']
    total_clusters = conn.execute("SELECT COUNT(*) as n FROM Clusters").fetchone()['n']
    total_tags     = conn.execute("SELECT COUNT(*) as n FROM DocumentTags").fetchone()['n']
    conn.close()
    return jsonify({'total_docs': total_docs, 'total_clusters': total_clusters, 'total_tags': total_tags})

@app.route('/api/assign', methods=['POST'])
def api_assign():
    data = request.get_json()
    doc_id = data.get('doc_id')
    cluster_id = data.get('cluster_id')
    conn = get_db()
    try:
        conn.execute("INSERT OR IGNORE INTO DocumentClusters VALUES (?,?)", (doc_id, cluster_id))
        conn.commit()
        conn.close()
        return jsonify({'status': 'ok'})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    init_db()
    app.run(debug=True)