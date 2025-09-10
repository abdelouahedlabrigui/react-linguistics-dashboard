# News API Documentation

This document provides an overview of the APIs available in `app_v1.py` and `app_v2.py`.

## Table of Contents

- [app_v1.py](#app_v1py)
  - [Endpoints](#endpoints)
- [app_v2.py](#app_v2py)
  - [Endpoints](#endpoints-1)
  - [Dependencies](#dependencies)
  - [Environment Variables](#environment-variables)
  - [Elasticsearch](#elasticsearch)
  - [Neo4j](#neo4j)
  - [OracleDB](#oracledb)
  - [Example Usage](#example-usage)

## app_v1.py

This file contains API endpoints for news search and retrieval using a Cypher graph and a relational database.

### Endpoints

-   **/news\_search\_retrieve\_actions\_by\_query**

    -   **Description:** Retrieves actions related to a news query using a Cypher graph.
    -   **Method:** GET
    -   **Parameters:**
        -   `query` (string, required): The search query (URL-encoded).
    -   **Response:**
        -   **Success (200):** Returns a JSON object containing the search results from the Cypher graph.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.

    -   **Example:** `/news_search_retrieve_actions_by_query?query=example%20query`

-   **/news\_search\_by\_query**

    -   **Description:** Searches news articles in a relational database based on a query.
    -   **Method:** GET
    -   **Parameters:**
        -   `query` (string, required): The search query (URL-encoded).
    -   **Response:**
        -   **Success (200):** Returns a JSON object containing the search results from the database.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.
    -   **Example:** `/news_search_by_query?query=example%20query`

## app_v2.py

This file contains API endpoints for news analysis using linguistics, psycholinguistics, Elasticsearch, and Neo4j.

### Endpoints

-   **/news/linguistics/v1**

    -   **Description:** Analyzes a news article using linguistics and stores the analysis in Elasticsearch.
    -   **Method:** POST
    -   **Request Body:** JSON object containing the news article with the following keys: `source`, `author`, `title`, `description`, `url`, `urlToImage`, `publishedAt`, `content`.
    -   **Response:**
        -   **Success (200):** Returns a JSON object with a "status" of "success" and a message indicating the number of documents in the Elasticsearch index.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.

    -   **Example:**

        ```json
        {
            "source": "Example Source",
            "author": "John Doe",
            "title": "Example Title",
            "description": "Example description of the article.",
            "url": "http://example.com/article",
            "urlToImage": "http://example.com/image.jpg",
            "publishedAt": "2024-01-01T00:00:00Z",
            "content": "This is the content of the example article."
        }
        ```

-   **/news/linguistics/v1/search**

    -   **Description:** Searches linguistic analysis data in Elasticsearch.
    -   **Method:** GET
    -   **Parameters:**
        -   `query` (string, required): The search query (URL-encoded).  Minimum length is 3 characters.
    -   **Response:**
        -   **Success (200):** Returns a JSON object containing the search results from Elasticsearch.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.
    -   **Example:** `/news/linguistics/v1/search?query=example%20search`

-   **/news/psycho\_linguistics/v1**

    -   **Description:** Analyzes a news article using psycholinguistics and stores the analysis in Elasticsearch.
    -   **Method:** POST
    -   **Request Body:** JSON object containing the news article with the following keys: `source`, `author`, `title`, `description`, `url`, `urlToImage`, `publishedAt`, `content`.
    -   **Response:**
        -   **Success (200):** Returns a JSON object with a "status" of "success" and a message indicating the number of documents in the Elasticsearch index.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.
    -   **Example:**

        ```json
        {
            "source": "Example Source",
            "author": "John Doe",
            "title": "Example Title",
            "description": "Example description of the article.",
            "url": "http://example.com/article",
            "urlToImage": "http://example.com/image.jpg",
            "publishedAt": "2024-01-01T00:00:00Z",
            "content": "This is the content of the example article."
        }
        ```

-   **/news/psycho\_linguistics/v1/search**

    -   **Description:** Searches psycho-linguistic analysis data in Elasticsearch.
    -   **Method:** GET
    -   **Parameters:**
        -   `query` (string, required): The search query (URL-encoded). Minimum length is 3 characters.
    -   **Response:**
        -   **Success (200):** Returns a JSON object containing the search results from Elasticsearch.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.
    -   **Example:** `/news/psycho_linguistics/v1/search?query=example%20search`

-   **/news/linguistics\_analyze\_relationships/v1**

    -   **Description:** Analyzes linguistic relationships using Neo4j and Elasticsearch.
    -   **Method:** POST
    -   **Request Body:** JSON object containing the linguistics analysis ID: `linguistics_analysis_id`.
    -   **Response:**
        -   **Success (200):** Returns a JSON object with a "status" of "success" and a message indicating the number of documents in the Elasticsearch index.
        -   **Error:** Returns a JSON object with a "status" of "error" and an error message.
    -   **Example:**

        ```json
        {
            "linguistics_analysis_id": "some_id"
        }
        ```

### Dependencies

-   Flask
-   elasticsearch8
-   pandas
-   requests
-   bs4 (BeautifulSoup4)
-   oracledb
-   flask\_cors
-   neo4j

### Environment Variables

The following environment variables should be set:

-   `ELASTICSEARCH_HOST`: The host address for Elasticsearch.  Currently hardcoded to `"http://10.42.0.243:9200"`
-   `NEO4J_URI`: The URI for the Neo4j database. Currently hardcoded to `"bolt://10.42.0.243:7687"`
-   `NEO4J_USER`: The username for the Neo4j database. Currently hardcoded to `"neo4j"`
-   `NEO4J_PASSWORD`: The password for the Neo4j database. Currently hardcoded to `"rootroot"`

### Elasticsearch

-   The code interacts with Elasticsearch for linguistic and psycholinguistic analysis.  Ensure Elasticsearch is running and accessible.

### Neo4j

-   The code interacts with Neo4j for analyzing relationships between linguistic elements. Ensure Neo4j is running and accessible.

### OracleDB

-   The code includes `oracledb`, but it is not actively used in the provided code.  It may be used in other parts of the application.

### Example Usage

1.  **Run the Flask application:**

    ```bash
    python app_v2.py
    ```

2.  **Send a POST request to /news/linguistics/v1:**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
        "source": "Example Source",
        "author": "John Doe",
        "title": "Example Title",
        "description": "Example description of the article.",
        "url": "http://example.com/article",
        "urlToImage": "http://example.com/image.jpg",
        "publishedAt": "2024-01-01T00:00:00Z",
        "content": "This is the content of the example article."
    }' http://localhost:5001/news/linguistics/v1
    ```

3.  **Send a GET request to /news/linguistics/v1/search:**

    ```bash
    curl "http://localhost:5001/news/linguistics/v1/search?query=example%20search"
    ```
