1. Web Scraping

The project uses the requests library to download HTML pages and BeautifulSoup to parse the HTML content.

The scraping process covers at least 3 different book categories and collects at least 60 books.

Fields Scraped

For every book, the following fields are collected:

Field	Description
title	Title of the book
price	Original price as listed on the website
star_rating	Rating as text such as One, Two, Three, Four, or Five
availability	Availability text shown on the website
category	Category of the book

Example:

title
price
star_rating
availability
category
2. Data Cleaning

After scraping, the raw data is cleaned before being inserted into the database.

Price Cleaning

The currency symbol is removed from the scraped price and the value is converted to a floating-point number.

For example:

£51.77

becomes:

51.77

The cleaned column is:

price_gbp

Example:

df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)
Star Rating Conversion

The website provides ratings as text:

One
Two
Three
Four
Five

These values are converted to integers:

Star Rating	Numeric Rating
One	1
Two	2
Three	3
Four	4
Five	5

The final column is:

rating

Example:

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)
Availability Conversion

The scraped availability text is converted into a Boolean column called:

in_stock

For example:

In stock

becomes:

True

If the availability indicates that the product is not available, it becomes:

False

Example:

df["in_stock"] = df["availability"].str.contains(
    "In stock",
    case=False,
    na=False
)
Handling Invalid Values

Web-scraped data can contain unexpected or missing values.

The pipeline is designed so that a single malformed row does not cause the entire pipeline to fail.

For numeric fields such as:

price_gbp
rating

invalid values are converted to missing values and replaced using the median of the corresponding column.

Example:

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

df["price_gbp"] = df["price_gbp"].fillna(
    df["price_gbp"].median()
)

For rating:

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

df["rating"] = df["rating"].fillna(
    df["rating"].median()
).astype(int)

This approach prevents the pipeline from crashing because of individual messy records while retaining as many scraped books as possible.

3. GBP to INR Conversion

The project uses the required fixed baseline exchange rate:

1 GBP = 105.50 INR

This is an artificial, project-defined constant for this assignment.

It is not a live market exchange rate and therefore no currency API or external lookup is required.

The INR price is calculated as:

price_inr = price_gbp × 105.50

Example:

GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

Therefore:

£10.00 × 105.50 = ₹1055.00

The fixed rate of 1 GBP = 105.50 INR is the required baseline used throughout this project.

4. Normalized SQLite Database

The cleaned data is stored in a SQLite database.

Database name:

books.db

The database uses a normalized two-table design.

Categories Table
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

The categories table stores each unique book category only once.

Books Table
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
);

The books table stores information about each book.

The category_id column creates the relationship between the two tables.

Database Relationship
categories
    |
    | category_id
    |
    ↓
books
    |
    └── category_id (Foreign Key)

Relationship:

categories.category_id
        ↓
books.category_id

This avoids storing the same category name repeatedly for every book.

5. Inserting Data into SQLite

Python's sqlite3 library is used to create the database and insert the cleaned data.

Example:

import sqlite3

connection = sqlite3.connect("books.db")

cursor = connection.cursor()

The tables are created using SQL:

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

The cleaned data is then inserted into the appropriate tables.

Finally:

connection.commit()
6. SQL Queries

At least five SQL queries are executed against the SQLite database.

The queries collectively demonstrate:

SELECT
WHERE
ORDER BY
LIMIT
DISTINCT
IN or BETWEEN
JOIN
Query 1: SELECT and WHERE

Find books costing more than £20.

SELECT title, price_gbp, rating
FROM books
WHERE price_gbp > 20;

This demonstrates:

SELECT
WHERE
Query 2: ORDER BY

Find books ordered from highest to lowest price.

SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC;

This demonstrates:

ORDER BY
Query 3: ORDER BY and LIMIT

Find the 10 most expensive books.

SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;

This demonstrates:

ORDER BY
LIMIT
Query 4: DISTINCT

Display all unique categories.

SELECT DISTINCT category_name
FROM categories;

This demonstrates:

DISTINCT
Query 5: BETWEEN

Find books whose prices are between £20 and £40.

SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40;

This demonstrates:

BETWEEN
Query 6: JOIN

List books together with their category names.

SELECT
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC
LIMIT 10;

This demonstrates a relational JOIN between:

books
categories

The query returns the 10 highest-rated books along with their categories.

7. Saving Query Results

Each SQL query and its resulting output are saved/displayed in the notebook.

Example:

query = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp > 20
"""

result = pd.read_sql(query, connection)

print(result)

The query string and DataFrame output are retained as part of the project results.

8. Reading SQL Results into Pandas

At least two SQL query results are read back into Pandas using:

pd.read_sql()

Example:

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp > 20
"""

df_query1 = pd.read_sql(query1, connection)

Another example:

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

df_query2 = pd.read_sql(query2, connection)

This demonstrates how SQL database results can be transferred back into Pandas for analysis.

9. Reproducing the JOIN Using Pandas

The SQL JOIN is also reproduced using pd.merge() directly on the in-memory DataFrames.

First, the category and book data are kept as separate DataFrames.

Example:

books_df
categories_df

The two DataFrames are merged using:

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

The result is equivalent to the SQL JOIN:

SELECT
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id;

The Pandas result is then sorted and limited to reproduce the SQL result.

Example:

pandas_join_result = (
    merged_df
    .sort_values("rating", ascending=False)
    .head(10)
)
10. SQL JOIN vs Pandas Merge

The project demonstrates the same relational operation in two ways.

SQL
books
JOIN
categories
ON books.category_id = categories.category_id
Pandas
pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

Both approaches should produce equivalent results when the same filtering, sorting, and limiting operations are applied.
