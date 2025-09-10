import oracledb
import logging
logger = logging.getLogger(__name__)

class SelectTables:
    def __init__(self):
        self.user = "SYS"
        self.password = "oracle"
        self.dsn = "10.42.0.243:1521/FREE"
    

    def filter_news_api_table_by_search(self, query: str) -> list:
        """Filter News Stocks Prompts Oracle Database Table by Title or Tickers"""
        try:
            connection = oracledb.connect(
                user=self.user, password=self.password, dsn=self.dsn, mode=oracledb.SYSDBA
            )
            cursor = connection.cursor()
            
            # SQL Query to filter table
            cursor.execute(
                f"""
                SELECT
                    ID, SOURCE, AUTHOR, Title, DESCRIPTION,
                    URL, URLTOIMAGE, PUBLISHEDAT, CONTENT
                FROM news_api_articles
                WHERE LOWER(DESCRIPTION) LIKE '%{query}%' OR LOWER(Title) LIKE '%{query}%' 
                ORDER BY ID ASC  
                """
            )
            rows = cursor.fetchall()  # Fetch all rows efficiently
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "source": row[1],
                    "author": row[2],
                    "title": row[3],
                    "description": row[4],
                    "url": row[5],
                    "urlToImage": row[6],
                    "publishedAt": row[7],
                    "content": row[8]
                })
            return results
        except Exception as e:
            logger.warning(f"Exception error: {e}")
            return []
        except oracledb.Error as e:
            logger.warning(f"Database exception error: {e}")
            return []
        
    