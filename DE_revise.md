**Key Observations from Past Papers:**

*   **Spark RDDs (Chapter 3):**
    *   Heavy emphasis on practical PySpark RDD coding.
    *   Questions involve:
        *   Creating RDDs from text files (`sc.textFile`).
        *   Common transformations: `flatMap` (for tokenization), `map` (to create pairs), `reduceByKey` (for counting/aggregation), `filter`, `distinct`, `sortBy`.
        *   Actions: `collect()` (to display results).
    *   Understanding of Pair RDDs is crucial for tasks like word count.
    *   The exam often provides an "Appendix A PySpark API Documentation" listing relevant methods – you need to know *how* and *when* to use them.
*   **MapReduce (Chapter 6):**
    *   Conceptual understanding: Explaining types of jobs (single mapper, multiple mappers reducer), stages (input, splitting, mapping, shuffling, reducing, output).
    *   Diagrammatic representation of MapReduce flow for specific tasks (e.g., word count, logistic company selection).
    *   Role of components like Record Reader and Combiner.
*   **Data Streaming & Kafka (Chapter 5):**
    *   Conceptual understanding: Differentiating batch vs. stream processing, explaining event streaming, replication.
    *   Kafka concepts: Roles of broker, producer, consumer.
    *   Applying Kafka to scenarios: Identifying topics, message contents, subscribers.
*   **NoSQL Databases (Chapter 4):**
    *   Conceptual understanding: Differences between traditional RDBMS and NoSQL.
    *   Applying NoSQL types to scenarios (Key-value, Document, Graph).
    *   Graph database concepts: Nodes, edges, properties.
    *   NoSQL distribution models: Replication, Sharding.
    *   HBase: Shell commands for table creation, data insertion, retrieval, schema alteration (often with an Appendix of commands).
*   **Hadoop Security (Chapter 7):**
    *   Best practices (employee training).
    *   Kerberos concepts: Principal, Authentication Server, Ticket. Illustrating Kerberos operation.
*   **HDFS Commands (Appendix in some papers):**
    *   Practical application: `mkdir`, `put`, `cp`, `mv`, `cat`, `ls`, `rm`.
*   **Spark Resource Configuration (Chapter 3):**
    *   Conceptual understanding of Spark job submission steps.
    *   Calculating resource configurations (executors per node, memory per executor) based on given cluster specs and YARN reservations.

**Strategy for Revision Notes & Materials:**

1.  **Prioritize Practical Coding/Commands:** Deep dive into PySpark RDD examples similar to Q4 in the "October Examination" and "January 2025" (Data Science) papers. Also, cover HDFS and HBase shell commands as seen in "November 2024" Q2 and "January 2025" (Business Analytics) Q4a.
2.  **Reinforce Core Concepts:** For MapReduce, Streaming, NoSQL types, and Security, focus on the definitions, roles, and applications frequently asked conceptually.
3.  **Diagrams:** Be prepared to draw diagrams for MapReduce workflows and potentially Kafka/NoSQL distribution.
4.  **Appendices are Key:** The exam often provides API lists. Your revision should make you comfortable quickly identifying and applying the correct function/command from such a list.

Let's start with **Chapter 3: Spark (RDD Focus for Practical Questions)** and then integrate HDFS commands.

## Chapter 3: Spark (RDD Practical Focus) & HDFS Commands

### **Revision Notes (Spark RDDs - Exam Style)**

**1. Understanding Spark Context and RDD Creation**

*   **SparkContext (`sc`):**
    *   The entry point for Spark functionality when working with RDDs.
    *   Provided in exam snippets: `from pyspark.sql import SparkSession`, `spark = SparkSession.builder.getOrCreate()`, `sc = spark.sparkContext`.
*   **Creating RDDs:**
    *   **From a text file (most common in exams):**
        *   `lines_rdd = sc.textFile("path/to/your/data.txt")`
        *   The path can be local (if Spark is running locally for testing) or an HDFS path (e.g., `hdfs:///user/student/data.txt`).
        *   Each line in the text file becomes a separate string element in the RDD.
        *   *Exam Tip:* Question 4a (Oct Exam) asks to create `lines_rdd` from `data.txt`.
    *   **From a Python collection (for testing or smaller data):**
        *   `data = [1, 2, 3, 4, 5]`
        *   `rdd = sc.parallelize(data)`

**2. Common RDD Transformations (Exam Relevance)**

*   **`flatMap(func)`:**
    *   **Purpose:** Applies a function to each element, but the function returns an iterator (e.g., a list of words from a line). All these iterators are then "flattened" into a single RDD.
    *   **Use Case (Very Common):** Word tokenization.
    *   **Example (from Oct Exam Q4b):**
        ```python
        # lines_rdd contains lines of text
        words_rdd = lines_rdd.flatMap(lambda line: line.split(" "))
        ```
*   **`map(func)`:**
    *   **Purpose:** Applies a function to each element, returning a new RDD with one output element for each input element.
    *   **Use Case (Very Common):** Creating key-value pairs, e.g., `(word, 1)` for word count.
    *   **Example (for word count, following `flatMap`):**
        ```python
        # words_rdd contains individual words
        word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
        ```
*   **`reduceByKey(func)`:**
    *   **Purpose:** Operates on Pair RDDs (RDDs of key-value tuples). For each key, it aggregates all its values using an associative and commutative function.
    *   **Use Case (Very Common):** Summing counts for word count, aggregating other metrics per key.
    *   **Example (for word count):**
        ```python
        # word_pairs_rdd contains (word, 1)
        word_counts_rdd = word_pairs_rdd.reduceByKey(lambda count1, count2: count1 + count2)
        ```
*   **`filter(func)`:**
    *   **Purpose:** Returns a new RDD containing only the elements for which the function `func` returns `True`.
    *   **Use Case:** Selecting specific data, e.g., words containing a certain letter (Oct Exam Q4e).
    *   **Example:**
        ```python
        # words_rdd contains individual words
        p_words_rdd = words_rdd.filter(lambda word: 'p' in word)
        ```
*   **`distinct()`:**
    *   **Purpose:** Returns a new RDD containing the unique elements from the source RDD.
    *   **Use Case:** Finding unique words (Oct Exam Q4e implicitly requires this before filtering for 'p' if the goal is *unique words containing 'p'*).
    *   **Example:**
        ```python
        unique_words_rdd = words_rdd.distinct()
        ```
*   **`sortBy(keyfunc, ascending=True/False)`:**
    *   **Purpose:** Sorts an RDD. The `keyfunc` specifies what to sort by (e.g., the second element of a tuple for sorting by count).
    *   **Use Case:** Sorting word counts (Oct Exam Q4d).
    *   **Example (sorting word_counts_rdd by count descending):**
        ```python
        # word_counts_rdd contains (word, count)
        sorted_word_counts_rdd = word_counts_rdd.sortBy(lambda wc_tuple: wc_tuple[1], ascending=False)
        ```
        *Exam Tip:* The appendix often provides `sortBy(key_function, [ascending=True])`. Pay attention to the lambda function needed to extract the sort key (e.g., `lambda x: x[1]` for the second element of a tuple).

Okay, let's go through common PySpark RDD transformations with examples and formatted outputs.

**Assumptions for Examples:**

*   `sc` is your available `SparkContext`.
*   We'll use simple datasets for clarity.
*   Outputs shown are what `rdd.collect()` would typically return. The order of elements in an RDD is generally not guaranteed unless explicitly sorted, but for these small examples, the order often appears deterministic.

---

### **PySpark RDD Transformations with Examples and Outputs**

**1. `map(func)`**

*   **Description:** Applies a function to each element of the RDD and returns a new RDD consisting of the results.
*   **Example:** Squaring each number in an RDD.

    ```python
    data = [1, 2, 3, 4, 5]
    rdd = sc.parallelize(data)
    squared_rdd = rdd.map(lambda x: x * x)
    print(squared_rdd.collect())
    ```
*   **Output:**
    ```
    [1, 4, 9, 16, 25]
    ```

**2. `filter(func)`**

*   **Description:** Returns a new RDD containing only the elements for which the given function `func` returns `True`.
*   **Example:** Filtering out even numbers from an RDD.

    ```python
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    rdd = sc.parallelize(data)
    odd_numbers_rdd = rdd.filter(lambda x: x % 2 != 0)
    print(odd_numbers_rdd.collect())
    ```
*   **Output:**
    ```
    [1, 3, 5, 7]
    ```

**3. `flatMap(func)`**

*   **Description:** Similar to `map`, but each input item can be mapped to 0 or more output items (the function `func` should return a sequence/iterator). The resulting sequences are then "flattened" into a single RDD.
*   **Example:** Splitting lines of text into individual words.

    ```python
    data = ["hello world", "spark is fun", "hello spark"]
    rdd = sc.parallelize(data)
    words_rdd = rdd.flatMap(lambda line: line.split(" "))
    print(words_rdd.collect())
    ```
*   **Output:**
    ```
    ['hello', 'world', 'spark', 'is', 'fun', 'hello', 'spark']
    ```

**4. `distinct()`**

*   **Description:** Returns a new RDD containing the distinct elements from the source RDD.
*   **Example:** Finding unique words from the previous `words_rdd`.

    ```python
    # Assuming words_rdd from the flatMap example:
    # words_rdd contains ['hello', 'world', 'spark', 'is', 'fun', 'hello', 'spark']
    unique_words_rdd = words_rdd.distinct()
    print(unique_words_rdd.collect())
    ```
*   **Output (order might vary):**
    ```
    ['hello', 'world', 'spark', 'is', 'fun']
    ```

**5. `union(otherRDD)`**

*   **Description:** Returns a new RDD containing all elements from the source RDD and the argument RDD. Duplicates are included.
*   **Example:** Combining two RDDs of numbers.

    ```python
    rdd1 = sc.parallelize([1, 2, 3, 4])
    rdd2 = sc.parallelize([3, 4, 5, 6])
    unioned_rdd = rdd1.union(rdd2)
    print(unioned_rdd.collect())
    ```
*   **Output:**
    ```
    [1, 2, 3, 4, 3, 4, 5, 6]
    ```

**6. `intersection(otherRDD)`**

*   **Description:** Returns a new RDD containing only elements found in both the source RDD and the argument RDD. Duplicates are removed.
*   **Example:** Finding common numbers between two RDDs.

    ```python
    rdd1 = sc.parallelize([1, 2, 3, 4, 3])
    rdd2 = sc.parallelize([3, 4, 5, 6, 4])
    intersected_rdd = rdd1.intersection(rdd2)
    print(intersected_rdd.collect())
    ```
*   **Output (order might vary):**
    ```
    [3, 4]
    ```

**7. `subtract(otherRDD)`**

*   **Description:** Returns a new RDD containing elements from the source RDD that are not present in the argument RDD.
*   **Example:** Finding numbers in `rdd1` but not in `rdd2`.

    ```python
    rdd1 = sc.parallelize([1, 2, 3, 4, 5])
    rdd2 = sc.parallelize([3, 4])
    subtracted_rdd = rdd1.subtract(rdd2)
    print(subtracted_rdd.collect())
    ```
*   **Output (order might vary):**
    ```
    [1, 2, 5]
    ```

**8. `cartesian(otherRDD)`**

*   **Description:** Returns a new RDD of all possible pairs `(a, b)` where `a` is in the source RDD and `b` is in the argument RDD. **Can be very expensive for large RDDs.**
*   **Example:** Creating all pairs from two small RDDs.

    ```python
    rdd1 = sc.parallelize([1, 2])
    rdd2 = sc.parallelize(['a', 'b'])
    cartesian_rdd = rdd1.cartesian(rdd2)
    print(cartesian_rdd.collect())
    ```
*   **Output:**
    ```
    [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    ```

---

### **Pair RDD Transformations**
*(These operate on RDDs where elements are key-value tuples, e.g., `(key, value)`)*

**Setup for Pair RDD Examples:**
```python
data_pairs = [("apple", 1), ("banana", 2), ("apple", 3), ("orange", 4), ("banana", 5)]
pair_rdd = sc.parallelize(data_pairs)
```

**9. `reduceByKey(func)`**

*   **Description:** Groups values for each key in the RDD and then aggregates the values for each key using the given associative and commutative reduce function `func`.
*   **Example:** Summing values for each fruit.

    ```python
    # Using pair_rdd from setup
    sum_by_key_rdd = pair_rdd.reduceByKey(lambda val1, val2: val1 + val2)
    print(sum_by_key_rdd.collect())
    ```
*   **Output (order of tuples might vary):**
    ```
    [('apple', 4), ('banana', 7), ('orange', 4)]
    ```

**10. `groupByKey()`**

*   **Description:** Groups all values for each key in the RDD into a single sequence. The result is an RDD of `(key, Iterable<values>)`.
*   **Example:** Grouping all counts for each fruit.

    ```python
    # Using pair_rdd from setup
    grouped_rdd = pair_rdd.groupByKey()
    # To see the content properly, we map the iterable to a list
    result = grouped_rdd.map(lambda kv: (kv[0], list(kv[1]))).collect()
    print(result)
    ```
*   **Output (order of tuples might vary, order within lists is preserved from original partitions):**
    ```
    [('apple', [1, 3]), ('banana', [2, 5]), ('orange', [4])]
    ```
    *(Note: `reduceByKey` is generally preferred over `groupByKey().map(...)` for aggregations due to better performance as `reduceByKey` does map-side combining.)*

**11. `sortByKey(ascending=True, numPartitions=None, keyfunc=lambda x: x)`**

*   **Description:** Sorts a Pair RDD by key.
*   **Example:** Sorting fruits alphabetically.

    ```python
    # Using pair_rdd from setup (or sum_by_key_rdd for a cleaner sort example)
    sum_by_key_rdd = pair_rdd.reduceByKey(lambda x,y: x+y) # from previous example
    sorted_rdd = sum_by_key_rdd.sortByKey(ascending=True)
    print(sorted_rdd.collect())
    ```
*   **Output:**
    ```
    [('apple', 4), ('banana', 7), ('orange', 4)]
    ```
    If sorting `sum_by_key_rdd.map(lambda x: (x[1], x[0])).sortByKey(ascending=False)` (sort by count descending):
    ```python
    # Using sum_by_key_rdd from reduceByKey example: [('apple', 4), ('banana', 7), ('orange', 4)]
    # Swap key-value to sort by count, then sort
    rdd_to_sort_by_value = sum_by_key_rdd.map(lambda kv: (kv[1], kv[0]))
    sorted_by_value_desc_rdd = rdd_to_sort_by_value.sortByKey(ascending=False)
    # Optionally map back to (fruit, count)
    final_sorted_rdd = sorted_by_value_desc_rdd.map(lambda kv: (kv[1], kv[0]))
    print(final_sorted_rdd.collect())
    ```
*   **Output for sorting by value descending:**
    ```
    [('banana', 7), ('apple', 4), ('orange', 4)]
    ```
    *(Or `[('banana', 7), ('orange', 4), ('apple', 4)]` - order of equal counts is not guaranteed without a secondary sort key)*

**12. `join(otherPairRDD)`**

*   **Description:** Performs an inner join between two Pair RDDs based on their keys. Returns an RDD of `(key, (value_from_rdd1, value_from_rdd2))`.
*   **Example:** Joining fruit prices with fruit stock.

    ```python
    prices_data = [("apple", 0.50), ("banana", 0.25), ("orange", 0.75)]
    stock_data = [("apple", 100), ("banana", 150), ("grape", 50)]
    prices_rdd = sc.parallelize(prices_data)
    stock_rdd = sc.parallelize(stock_data)

    joined_rdd = prices_rdd.join(stock_rdd)
    print(joined_rdd.collect())
    ```
*   **Output (order of tuples might vary):**
    ```
    [('apple', (0.5, 100)), ('banana', (0.25, 150))]
    ```

**13. `leftOuterJoin(otherPairRDD)`**

*   **Description:** Performs a left outer join. For each key `k` in the source RDD, the resulting RDD will contain `(k, (v, w))` where `v` is the value from the source RDD and `w` is an `Option(value)` from `otherPairRDD` (`None` if no match).
*   **Example:**

    ```python
    # Using prices_rdd and stock_rdd from join example
    left_joined_rdd = prices_rdd.leftOuterJoin(stock_rdd)
    print(left_joined_rdd.collect())
    ```
*   **Output (order of tuples might vary):**
    ```
    [('apple', (0.5, 100)), ('banana', (0.25, 150)), ('orange', (0.75, None))]
    ```

**14. `rightOuterJoin(otherPairRDD)`**

*   **Description:** Performs a right outer join.
*   **Example:**

    ```python
    # Using prices_rdd and stock_rdd from join example
    right_joined_rdd = prices_rdd.rightOuterJoin(stock_rdd)
    print(right_joined_rdd.collect())
    ```
*   **Output (order of tuples might vary):**
    ```
    [('apple', (0.5, 100)), ('banana', (0.25, 150)), ('grape', (None, 50))]
    ```

**15. `cogroup(otherPairRDD)`**

*   **Description:** For each key `k` in `this` or `otherPairRDD`, return a resulting RDD that contains a tuple with the list of values for that key in `this` as well as `otherPairRDD`. The result is `(key, (Iterable<V>, Iterable<W>))`.
*   **Example:**

    ```python
    rdd1 = sc.parallelize([("a", 1), ("a", 2), ("b", 3)])
    rdd2 = sc.parallelize([("a", 4), ("c", 5)])
    cogrouped_rdd = rdd1.cogroup(rdd2)
    # Map to list for easier viewing
    result = cogrouped_rdd.map(lambda x: (x[0], (list(x[1][0]), list(x[1][1])))).collect()
    print(result)
    ```
*   **Output (order of tuples might vary):**
    ```
    [('a', ([1, 2], [4])), ('b', ([3], [])), ('c', ([], [5]))]
    ```

**16. `mapValues(func)`**

*   **Description:** Applies a function only to the values of a Pair RDD, leaving the keys unchanged. This is more efficient than `map` if only values need transformation because it avoids shuffling keys.
*   **Example:** Incrementing the count for each fruit.

    ```python
    # Using sum_by_key_rdd from reduceByKey example: [('apple', 4), ('banana', 7), ('orange', 4)]
    incremented_values_rdd = sum_by_key_rdd.mapValues(lambda count: count + 10)
    print(incremented_values_rdd.collect())
    ```
*   **Output (order of tuples might vary):**
    ```
    [('apple', 14), ('banana', 17), ('orange', 14)]
    ```

**17. `keys()` and `values()`**

*   **Description:**
    *   `keys()`: Returns an RDD of only the keys from a Pair RDD.
    *   `values()`: Returns an RDD of only the values from a Pair RDD.
*   **Example:**

    ```python
    # Using sum_by_key_rdd from reduceByKey example: [('apple', 4), ('banana', 7), ('orange', 4)]
    fruit_keys_rdd = sum_by_key_rdd.keys()
    fruit_values_rdd = sum_by_key_rdd.values()
    print("Keys:", fruit_keys_rdd.collect())
    print("Values:", fruit_values_rdd.collect())
    ```
*   **Output:**
    ```
    Keys: ['apple', 'banana', 'orange']
    Values: [4, 7, 4]
    ```
    *(Order for keys and values will correspond to the order in `sum_by_key_rdd` before `collect()`)*

---

This list covers many of the common RDD transformations. Remember that transformations are lazy and only execute when an action is called. The appendices in your exam papers are crucial for knowing which specific methods are considered "in scope."
        

**3. RDD Actions (Exam Relevance)**

*   **`collect()`:**
    *   **Purpose:** Returns all elements of the RDD as a list to the driver program.
    *   **Use Case:** Displaying results in the exam.
    *   **Caution:** Can cause OutOfMemoryError on the driver if the RDD is very large. Exams usually use small datasets where `collect()` is fine.
    *   **Example:** `print(word_counts_rdd.collect())`
*   **`take(n)`:** Returns the first `n` elements.
*   **`count()`:** Returns the number of elements.
*   **`first()`:** Returns the first element.

**4. HDFS Commands (from Nov 2024 Exam Q2 & Appendix)**

*   **Context:** Assume you are logged in as user `student` locally, and Hadoop user is `hduser`. HDFS paths often relative to `/user/hduser/`.
*   **`hdfs dfs -mkdir <hdfspath>`:** Creates a directory in HDFS.
    *   Example (Nov Q2a): `hdfs dfs -mkdir /user/hduser/data` (If `hduser`'s home is `/user/hduser`) or simply `hdfs dfs -mkdir data` (if current HDFS dir is `/user/hduser`).
*   **`hdfs dfs -put <localsrc> <dest>`:** Copies file/dir from local filesystem to HDFS.
    *   Example (Nov Q2b): `hdfs dfs -put first_file.txt /user/hduser/data/`
*   **`hdfs dfs -cp <source> <destination>`:** Copies file/dir within HDFS (or from local if source is local).
    *   Example (Nov Q2c): `hdfs dfs -cp /user/hduser/data/first_file.txt /user/hduser/data/copy_of_first_file.txt`
*   **`hdfs dfs -mv <src> <dest>`:** Moves/renames file/dir within HDFS.
    *   Example (Nov Q2d): `hdfs dfs -mv /user/hduser/data/copy_of_first_file.txt /user/hduser/data/second_file.txt`
*   **`hdfs dfs -cat <hdfs-file-path>`:** Displays contents of an HDFS file.
    *   Example (Nov Q2e): `hdfs dfs -cat /user/hduser/data/second_file.txt`
*   **`hdfs dfs -get <hdfspath> <localfile>`:** Copies file/dir from HDFS to local filesystem.
    *   Example (Nov Q2f): `hdfs dfs -get /user/hduser/data/second_file.txt .` (copies to current local dir)
*   **`hdfs dfs -rm <hdfspaths>`:** Deletes files. Use `-r` or `-R` for recursive delete of directories.
    *   Example (Nov Q2g): `hdfs dfs -rm /user/hduser/data/first_file.txt`
    *   Example (Nov Q2i, delete directory): `hdfs dfs -rm -r /user/hduser/data`
*   **`hdfs dfs -ls <hdfspath>`:** Lists files and directories in an HDFS path.
    *   Example (Nov Q2h): `hdfs dfs -ls /user/hduser/data`

---

### **Spark RDD & HDFS Flashcards**

**Questions:**

**Spark RDDs:**
1.  What is the PySpark RDD method to create an RDD from `data.txt` located in HDFS?
2.  Given `lines_rdd = sc.textFile("file.txt")`, how do you split each line into words and get a flat list of all words? (Transformation name)
3.  After getting `words_rdd`, how do you create key-value pairs where each word is a key and its value is 1? (Transformation name)
4.  If `word_pairs_rdd` contains `[('hello', 1), ('world', 1), ('hello', 1)]`, what RDD transformation sums the values for each key?
5.  What is the PySpark RDD API method to get unique elements from an RDD?
6.  How do you sort `word_count_rdd = [('world', 1), ('hello', 2)]` by count (the second element) in descending order?
7.  Which RDD *action* brings all elements of the RDD to the driver as a Python list?
8.  What is the RDD *transformation* `mapValues(func)` used for? (See Oct Exam Appendix)
9.  What is the RDD *action* `countByKey()` used for? (See Oct Exam Appendix)
10. If an exam provides `sortBy(key_function, [ascending=True])`, and you have `rdd = sc.parallelize([('b',1), ('a',2)])`, write the `key_function` to sort by the first element (the letter).

**HDFS Commands:**
11. HDFS command to create a directory named `output` inside `/user/hduser/project/`.
12. HDFS command to copy a local file `report.txt` to the HDFS directory `/user/hduser/reports/`.
13. HDFS command to display the content of `/user/hduser/logs/app.log` on the screen.
14. HDFS command to rename `/user/hduser/old_name.txt` to `/user/hduser/new_name.txt`.
15. HDFS command to delete the HDFS directory `/user/hduser/temp_data` and all its contents.

**Answers:**

**Spark RDDs:**
1.  `sc.textFile("hdfs:///path/to/data.txt")` (or just `sc.textFile("data.txt")` if HDFS path context is clear).
2.  `flatMap(lambda line: line.split(' '))`
3.  `map(lambda word: (word, 1))`
4.  `reduceByKey(lambda a, b: a + b)`
5.  `distinct()`
6.  `word_count_rdd.sortBy(lambda item: item[1], ascending=False)`
7.  `collect()`
8.  Passes each value in a key-value pair RDD through the given `map_function` without changing the keys; retains original RDD’s partitioning.
9.  Counts the number of elements for each key and returns the result to the master as a dictionary.
10. `lambda x: x[0]`

**HDFS Commands:**
11. `hdfs dfs -mkdir /user/hduser/project/output`
12. `hdfs dfs -put report.txt /user/hduser/reports/`
13. `hdfs dfs -cat /user/hduser/logs/app.log`
14. `hdfs dfs -mv /user/hduser/old_name.txt /user/hduser/new_name.txt`
15. `hdfs dfs -rm -R /user/hduser/temp_data` (or `-rm -r`)

---

### **Mini-Tests (Spark RDD & HDFS - Exam Style)**

**Mini-Test 1: PySpark RDD Coding**

**Scenario:** You are given a text file `reviews.txt` in HDFS at `/user/student/reviews.txt`. Each line contains a product review.

**Content of `reviews.txt` (example):**
```
Great product amazing quality
Bad product broke easily
Amazing service great price
```

**Provided SparkContext:** `sc`

**Tasks (write the PySpark RDD code for each):**

1.  Create an RDD named `review_lines_rdd` from `reviews.txt`. (1 mark)
2.  Tokenize `review_lines_rdd` into individual words, storing the result in `words_rdd`. (2 marks)
3.  Create a Pair RDD `word_pairs_rdd` where each word is a key and its value is 1. (2 marks)
4.  Calculate the count of each word and store it in `word_counts_rdd`. (2 marks)
5.  Filter `word_counts_rdd` to get only words that appear more than once, store in `frequent_words_rdd`. (2 marks)
6.  Display all elements of `frequent_words_rdd`. (1 mark)

---

**Mini-Test 2: HDFS Shell Commands**

**Scenario:** Your current local directory contains a file `sales_data.csv`. Your HDFS home directory is `/user/student/`.

**Tasks (write the HDFS command for each):**

1.  Create a new directory named `staging` inside your HDFS home directory. (1 mark)
2.  Copy `sales_data.csv` from your local directory into the HDFS `staging` directory. (1 mark)
3.  List the contents of the HDFS `staging` directory. (1 mark)
4.  Create a copy of `sales_data.csv` within the HDFS `staging` directory and name the copy `sales_data_backup.csv`. (1 mark)
5.  Display the first few lines of `sales_data.csv` from HDFS (assuming a command similar to `head` is available or how you'd use `cat` for this if `head` isn't in the appendix). (1 mark - *if `head` isn't provided in exam appendix, `cat` is acceptable with an explanation*)
6.  Delete the original `sales_data.csv` from the HDFS `staging` directory. (1 mark)
7.  Move `sales_data_backup.csv` from `/user/student/staging/` to a new HDFS directory `/user/student/archive/` (assume `archive` needs to be created first if it doesn't exist). (2 marks)

---
**Answer Key (Mini-Tests)**

**Mini-Test 1: PySpark RDD Coding**

1.  ```python
    review_lines_rdd = sc.textFile("/user/student/reviews.txt")
    ```
2.  ```python
    words_rdd = review_lines_rdd.flatMap(lambda line: line.lower().split(" ")) # .lower() is good practice
    ```
3.  ```python
    word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
    ```
4.  ```python
    word_counts_rdd = word_pairs_rdd.reduceByKey(lambda a, b: a + b)
    ```
5.  ```python
    frequent_words_rdd = word_counts_rdd.filter(lambda item: item[1] > 1)
    ```
6.  ```python
    print(frequent_words_rdd.collect())
    ```

**Mini-Test 2: HDFS Shell Commands**

1.  `hdfs dfs -mkdir staging` (or `hdfs dfs -mkdir /user/student/staging`)
2.  `hdfs dfs -put sales_data.csv staging/` (or `hdfs dfs -put sales_data.csv /user/student/staging/`)
3.  `hdfs dfs -ls staging` (or `hdfs dfs -ls /user/student/staging`)
4.  `hdfs dfs -cp staging/sales_data.csv staging/sales_data_backup.csv`
5.  `hdfs dfs -cat staging/sales_data.csv | head` (If `head` is allowed with pipe. If not, and only appendix commands are allowed: `hdfs dfs -cat staging/sales_data.csv`. *Explanation:* The `cat` command will display the entire file; to see only the first few lines typically requires an additional utility like `head`, which might not be directly part of HDFS commands but usable in a shell environment.)
6.  `hdfs dfs -rm staging/sales_data.csv`
7.  ```bash
    hdfs dfs -mkdir archive 
    hdfs dfs -mv staging/sales_data_backup.csv archive/
    ```
    (or `hdfs dfs -mkdir /user/student/archive` then `hdfs dfs -mv /user/student/staging/sales_data_backup.csv /user/student/archive/`)

---

### **Finals Example Questions (Mimicking Past Year Styles)**

**Question A (Similar to Nov 2024 Q2 / Jan 2025 (BA) Q2 - HDFS Commands)**

You have a local file named `transactions.log`. Your Hadoop username is `analyst01`.
Appendix A provides a list of HDFS commands. Write the HDFS commands to perform the following:

i.  Create a directory named `logs` directly under your HDFS user directory. (2 marks)
ii. Upload `transactions.log` from your local system into the newly created `logs` HDFS directory. (2 marks)
iii. Create a subdirectory named `archive` inside the HDFS `logs` directory. (2 marks)
iv. Move the `transactions.log` file from `/user/analyst01/logs/` to `/user/analyst01/logs/archive/`. (2 marks)
v.  List all files and directories within `/user/analyst01/logs/archive/`. (2 marks)

**(Assume Appendix A provides: `mkdir`, `put`, `mv`, `ls`, `rm`, `cat`, `cp`)**

**Question B (Similar to Oct Exam Q4 / Jan 2025 (DS) Q4 - PySpark RDDs)**

A text file named `feedback.txt` stored in HDFS contains customer feedback, one feedback entry per line.
The SparkContext `sc` is available. Appendix B provides PySpark RDD methods.

**Content of `feedback.txt` (example):**
```
Service was excellent and fast
Product quality good delivery slow
Fast response excellent support
```

Write PySpark RDD code to:

i.  Create an RDD named `feedback_rdd` from the text file `feedback.txt`. Then, display `feedback_rdd` as a list. (3 marks)
ii. Perform word tokenization on `feedback_rdd` (convert to lowercase first) and store the flattened results in a new RDD named `tokens_rdd`. (4 marks)
iii. Create a paired RDD named `token_counts_rdd` containing key-value pairs representing each token and its corresponding number of occurrences. (5 marks)
iv. Create an RDD named `sorted_token_counts_rdd` which sorts `token_counts_rdd` by the number of occurrences in *ascending* order. (4 marks)
v.  Create an RDD that contains only the unique tokens from `tokens_rdd` that have a length greater than 4 characters. Display the result as a list. (4 marks)

**(Assume Appendix B provides: `textFile`, `collect`, `flatMap`, `map`, `reduceByKey`, `sortBy`, `filter`, `distinct`)**

---

This focused revision should align well with the practical coding and command-line aspects highlighted in the past papers. Remember to also review the conceptual parts of Chapters 5, 6, and 7 from your lecture notes. Good luck!




Okay, let's create the revision notes, flashcards, and mini-tests for **Chapter 4 (NoSQL Databases, with a focus on shell commands from Appendix 4.1)** and **Chapter 5 (Data Streaming, focusing on concepts likely to appear based on past paper styles)**.

## Chapter 4: NoSQL Databases (Conceptual & Shell Command Focus)

### **Revision Notes (Chapter 4 - Conceptual & Exam Style Shell Commands)**

**1. Core Concepts & Differences (Exam Relevance: Nov 2024 Q3b, Jan 2025 (BA) Q2a)**

*   **Traditional Database Management Systems (RDBMS):**
    *   Structure: Relational (tables with rows and columns).
    *   Schema: Fixed, predefined (schema-on-write).
    *   Scalability: Primarily vertical (scale-up).
    *   Consistency: Strong (ACID properties).
    *   Data Model: Structured data.
    *   Query Language: SQL.
*   **NoSQL Databases:**
    *   Structure: Various models (Key-Value, Document, Column-Family, Graph). "Not Only SQL."
    *   Schema: Flexible, dynamic (schema-on-read or schema-less).
    *   Scalability: Primarily horizontal (scale-out).
    *   Consistency: Often eventual consistency (BASE properties).
    *   Data Model: Can handle unstructured, semi-structured, and structured data.
    *   Query Language: Varies by database type (e.g., MongoDB Query Language, Cypher).
*   **Main Difference (for exam answers):** RDBMS typically have rigid schemas, scale vertically, and enforce ACID properties, while NoSQL databases offer flexible schemas, scale horizontally, and often follow BASE consistency. NoSQL is designed for large volumes of rapidly changing, diverse data types.

**2. NoSQL Data Store Applications (Exam Relevance: Nov 2024 Q3c, Jan 2025 (BA) Q2a)**

*   **Key-Value Data Store:**
    *   **How it may be used (e.g., Movie Recommender / NLP Project):**
        *   **Caching:** Store frequently accessed data, like user profiles, pre-computed recommendations, or popular movie details. Key: `user_id:profile` or `movie_id:details`. Value: JSON string or serialized object.
        *   **Session Management:** Store user session data. Key: `session_id`. Value: User activity, preferences.
        *   **NLP Project:** Store word-to-ID mappings, or simple feature lookups. Key: `word`. Value: `word_id` or `sentiment_score`.
    *   **Illustration:** Simple key-value pairs. `user:123:fav_genre -> "Sci-Fi"`, `movie:tt0088763:title -> "Back to the Future"`.
*   **Document-Based Database:**
    *   **How it may be used (e.g., Movie Recommender / NLP Project):**
        *   **User Profiles:** Store complex user profiles with nested information (preferences, watch history, ratings) in a single document.
            *   Example: `{ "_id": "user123", "username": "john_doe", "preferences": {"genres": ["Sci-Fi", "Action"], "actors": ["Harrison Ford"]}, "watch_history": [{"movie_id": "m001", "rating": 5, "timestamp": "..."}] }`
        *   **Movie Details:** Store all information about a movie (title, cast, crew, reviews, synopsis) as a single document.
        *   **NLP Project:** Store processed documents with their metadata, annotations, and extracted features as a single JSON-like document.
    *   **Solves problem of different data rows having different columns (vs. RDBMS):** Document databases allow each document in a collection to have its own unique structure. New fields can be added to some documents without affecting others. This is ideal for evolving data or data where not all entities have the same attributes (e.g., some movies have sequels, others don't).

**3. NoSQL Distribution Models (Exam Relevance: Jan 2025 (BA) Q2b)**

*   **Replication:**
    *   **Concept:** Copying the same data across multiple nodes (servers/replicas).
    *   **Purpose:**
        *   **High Availability & Fault Tolerance:** If one node fails, data is still accessible from other replicas.
        *   **Read Scalability:** Read requests can be distributed across multiple replicas, improving read throughput.
    *   **Techniques (Diagrams):**
        *   **Master-Slave Replication:** One primary node (master) handles all writes. Writes are then propagated to secondary nodes (slaves). Slaves typically handle read requests.
            *   *Diagram:* Central master node with arrows pointing to several slave nodes. Write operations go to master, read operations can go to slaves.
        *   **Peer-to-Peer (or Masterless) Replication:** All nodes can accept reads and writes. Data changes are propagated to other peers. More complex to manage consistency.
            *   *Diagram:* Multiple nodes connected in a mesh or ring, with arrows indicating data can flow between any pair for reads/writes.
*   **Sharding (Partitioning):**
    *   **Concept:** Distributing different subsets of data across multiple nodes (shards). Each shard holds a unique portion of the total dataset.
    *   **Purpose:**
        *   **Write Scalability:** Write operations can be distributed across different shards, improving write throughput.
        *   **Storage Scalability:** Dataset size can exceed the capacity of a single server.
    *   **Techniques (Diagrams):**
        *   **Range-Based Sharding:** Data is partitioned based on a range of a shard key (e.g., User IDs 1-1000 on Shard A, 1001-2000 on Shard B).
            *   *Diagram:* Data items (e.g., A-M, N-Z) split into different server blocks.
        *   **Hash-Based Sharding:** A shard key is hashed, and the hash value determines which shard the data goes to. Distributes data more evenly but can make range queries difficult.
            *   *Diagram:* Data items passed through a hash function, then distributed to different server blocks based on hash output.
        *   **Directory-Based Sharding:** A lookup service (directory) maintains mapping of shard keys to physical shards.
    *   *Exam Tip for Diagrams:* Keep them simple. Show a dataset being divided and stored on different server icons. Clearly label the sharding key or method if possible.

**4. Shell Commands (from Appendix 4.1 - covered in previous response, ensure familiarity)**

*   **Redis:** `SET`, `GET`, `APPEND`, `STRLEN`, `RPUSH`, `LPOP`, `RPOP`, `LRANGE`, `HSET`, `HGET`, `HGETALL`, `SADD`, `SMEMBERS`, `SISMEMBER`, `TTL`, `EXPIRE`.
*   **MongoDB:** `use`, `db.createCollection`, `db.collection.drop`, `db.dropDatabase`, `db.collection.insertOne`, `db.collection.insertMany`, `db.collection.find`, `db.collection.updateMany` (with operators like `$set`).
    *   **MongoDB Operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`, `$not`.
*   **Neo4j Cypher:** `CREATE`, `MERGE`, `MATCH`, `RETURN`, `SET`.
*   **HBase:** `create`, `disable`, `enable`, `alter`, `describe`, `list`, `put`, `get`, `scan`, `count`.


Okay, here are example inputs and outputs for the NoSQL shell commands listed, categorized by the database system.

**Note:**
*   Outputs can sometimes vary slightly depending on the shell version or specific configuration.
*   For commands that modify data, the "Output" often shows a confirmation, and the "State After" (or implied state) shows the result of the operation.
*   Placeholders like `ObjectId("...")` represent system-generated IDs.
*   Timestamps in HBase output are omitted for brevity but are always present.

---

#### Redis

**Initial State:** Assume an empty Redis instance unless specified.

1.  **SET key value**
    *   **Command:** `SET mykey "Hello Redis"`
    *   **Example Input:** `SET user:1:name "Alice"`
    *   **Example Output:** `OK`
    *   **State After:** Key `user:1:name` stores the string value `"Alice"`.

2.  **GET key**
    *   **Command:** `GET mykey`
    *   **Example Input (key exists):** `GET user:1:name`
    *   **Example Output:** `"Alice"`
    *   **Example Input (key does not exist):** `GET user:2:name`
    *   **Example Output:** `(nil)`

3.  **APPEND key value**
    *   **Command:** `APPEND mykey " World"`
    *   **Example Input (key exists):**
        ```redis
        SET greeting "Hello"
        APPEND greeting " World"
        ```
    *   **Example Output (for APPEND):** `(integer) 11` (the new length of the string)
    *   **State After:** Key `greeting` stores `"Hello World"`.
    *   **Example Input (key does not exist):** `APPEND newmessage "Start"`
    *   **Example Output:** `(integer) 5`
    *   **State After:** Key `newmessage` stores `"Start"`.

4.  **STRLEN key**
    *   **Command:** `STRLEN mykey`
    *   **Example Input:** (Assuming `greeting` is `"Hello World"`)
        `STRLEN greeting`
    *   **Example Output:** `(integer) 11`

5.  **RPUSH key value1 [value2 ...]**
    *   **Command:** `RPUSH mylist "one" "two" "three"`
    *   **Example Input:** `RPUSH tasks "task1" "task2"`
    *   **Example Output:** `(integer) 2` (the new length of the list)
    *   **State After:** List `tasks` contains `["task1", "task2"]`.
    *   **Example Input (adding more):** `RPUSH tasks "task3"`
    *   **Example Output:** `(integer) 3`
    *   **State After:** List `tasks` contains `["task1", "task2", "task3"]`.

6.  **LPOP key**
    *   **Command:** `LPOP mylist`
    *   **Example Input:** (Assuming `tasks` is `["task1", "task2", "task3"]`)
        `LPOP tasks`
    *   **Example Output:** `"task1"`
    *   **State After:** List `tasks` contains `["task2", "task3"]`.
    *   **Example Input (empty list):** (Assuming `emptylist` does not exist or is empty)
        `LPOP emptylist`
    *   **Example Output:** `(nil)`

7.  **RPOP key**
    *   **Command:** `RPOP mylist`
    *   **Example Input:** (Assuming `tasks` is `["task2", "task3"]`)
        `RPOP tasks`
    *   **Example Output:** `"task3"`
    *   **State After:** List `tasks` contains `["task2"]`.

8.  **LRANGE key start stop**
    *   **Command:** `LRANGE mylist 0 -1`
    *   **Example Input:** (Assuming `tasks` is `["task2"]` and we add more: `RPUSH tasks "another" "final"`)
        `RPUSH tasks "another" "final"` (Output: `(integer) 3`)
        `LRANGE tasks 0 1`
    *   **Example Output (for LRANGE):**
        ```
        1) "task2"
        2) "another"
        ```
    *   **Example Input (get all):** `LRANGE tasks 0 -1`
    *   **Example Output:**
        ```
        1) "task2"
        2) "another"
        3) "final"
        ```

9.  **HSET key field value**
    *   **Command:** `HSET user:1 username "JohnDoe" email "john.doe@example.com"`
    *   **Example Input:** `HSET user:profile:100 name "Bob" age "25"`
    *   **Example Output:** `(integer) 2` (number of fields added/updated)
    *   **State After:** Hash `user:profile:100` contains fields `name: "Bob"` and `age: "25"`.

10. **HGET key field**
    *   **Command:** `HGET user:1 username`
    *   **Example Input:** `HGET user:profile:100 name`
    *   **Example Output:** `"Bob"`
    *   **Example Input (field does not exist):** `HGET user:profile:100 city`
    *   **Example Output:** `(nil)`

11. **HGETALL key**
    *   **Command:** `HGETALL user:1`
    *   **Example Input:** `HGETALL user:profile:100`
    *   **Example Output:**
        ```
        1) "name"
        2) "Bob"
        3) "age"
        4) "25"
        ```
        (An array of field names and values, interleaved)

12. **SADD key member [member ...]**
    *   **Command:** `SADD myset "apple" "banana" "cherry"`
    *   **Example Input:** `SADD user:100:tags "sports" "music" "movies" "music"`
    *   **Example Output:** `(integer) 3` (number of new members added; "music" was a duplicate)
    *   **State After:** Set `user:100:tags` contains `{"sports", "music", "movies"}`.

13. **SMEMBERS key**
    *   **Command:** `SMEMBERS myset`
    *   **Example Input:** `SMEMBERS user:100:tags`
    *   **Example Output:** (Order is not guaranteed)
        ```
        1) "music"
        2) "sports"
        3) "movies"
        ```

14. **SISMEMBER key member**
    *   **Command:** `SISMEMBER myset "apple"`
    *   **Example Input (member exists):** `SISMEMBER user:100:tags "music"`
    *   **Example Output:** `(integer) 1`
    *   **Example Input (member does not exist):** `SISMEMBER user:100:tags "travel"`
    *   **Example Output:** `(integer) 0`

15. **TTL key**
    *   **Command:** `TTL mykey`
    *   **Example Input (key with expiration):**
        ```redis
        SET tempkey "will expire"
        EXPIRE tempkey 60
        TTL tempkey
        ```
    *   **Example Output (for TTL):** `(integer) 59` (or slightly less, seconds remaining)
    *   **Example Input (key without expiration):**
        ```redis
        SET permkey "permanent"
        TTL permkey
        ```
    *   **Example Output (for TTL):** `(integer) -1`
    *   **Example Input (key does not exist):** `TTL non_existent_key`
    *   **Example Output:** `(integer) -2`

16. **EXPIRE key seconds**
    *   **Command:** `EXPIRE mykey 30`
    *   **Example Input:**
        ```redis
        SET session:123 "userdata"
        EXPIRE session:123 3600
        ```
    *   **Example Output (for EXPIRE):** `(integer) 1` (1 if timeout was set, 0 if key doesn't exist)
    *   **State After:** `session:123` will be deleted after 3600 seconds.

---

#### MongoDB

**Initial State:** Assume a MongoDB instance is running. `>` indicates the mongo shell prompt.

1.  **use database_name**
    *   **Command:** `use mydatabase`
    *   **Example Input:** `> use storeDB`
    *   **Example Output:** `switched to db storeDB`
    *   **State After:** The current database context is set to `storeDB`. If it didn't exist, it's created upon first data insertion.

2.  **db.createCollection("collection_name")**
    *   **Command:** `db.createCollection("products")`
    *   **Example Input:** `> db.createCollection("customers")`
    *   **Example Output:** `{ "ok" : 1 }`
    *   **State After:** An empty collection named `customers` is created in the current database (`storeDB`).

3.  **db.collection_name.drop()**
    *   **Command:** `db.products.drop()`
    *   **Example Input:** `> db.customers.drop()`
    *   **Example Output:** `true`
    *   **State After:** The `customers` collection and all its documents are removed.

4.  **db.dropDatabase()**
    *   **Command:** `db.dropDatabase()`
    *   **Example Input:** (Assuming current DB is `storeDB`)
        `> db.dropDatabase()`
    *   **Example Output:** `{ "dropped" : "storeDB", "ok" : 1 }`
    *   **State After:** The entire `storeDB` database is deleted.

5.  **db.collection_name.insertOne({ key: value })**
    *   **Command:** `db.users.insertOne({ name: "Alice", age: 30, city: "New York" })`
    *   **Example Input:** (Assuming `use storeDB` and `db.createCollection("products")` ran)
        `> db.products.insertOne({ name: "Laptop", price: 1200, category: "Electronics" })`
    *   **Example Output:**
        ```json
        {
          "acknowledged" : true,
          "insertedId" : ObjectId("someGeneratedIdValue")
        }
        ```
    *   **State After:** A new document is added to the `products` collection.

6.  **db.collection_name.insertMany([...])**
    *   **Command:** `db.users.insertMany([{ name: "Bob", age: 24 }, { name: "Charlie", age: 35 }])`
    *   **Example Input:**
        `> db.products.insertMany([
        ...   { name: "Mouse", price: 25, category: "Electronics" },
        ...   { name: "Keyboard", price: 75, category: "Electronics" }
        ... ])`
    *   **Example Output:**
        ```json
        {
          "acknowledged" : true,
          "insertedIds" : [
            ObjectId("anotherGeneratedId1"),
            ObjectId("anotherGeneratedId2")
          ]
        }
        ```
    *   **State After:** Two new documents are added to the `products` collection.

7.  **db.collection_name.find({ key: value })**
    *   **Command:** `db.users.find({ city: "New York" })`
    *   **Example Input:** `> db.products.find({ category: "Electronics" })`
    *   **Example Output:** (Will list all matching documents, formatted)
        ```json
        { "_id" : ObjectId("someGeneratedIdValue"), "name" : "Laptop", "price" : 1200, "category" : "Electronics" }
        { "_id" : ObjectId("anotherGeneratedId1"), "name" : "Mouse", "price" : 25, "category" : "Electronics" }
        { "_id" : ObjectId("anotherGeneratedId2"), "name" : "Keyboard", "price" : 75, "category" : "Electronics" }
        ```

8.  **db.collection_name.updateMany({ query_field: query_value }, { $set: { update_field: update_value } })**
    *   **Command:** `db.users.updateMany({ city: "New York" }, { $set: { country: "USA" } })`
    *   **Example Input:** `> db.products.updateMany({ category: "Electronics" }, { $set: { inStock: true } })`
    *   **Example Output:**
        ```json
        { "acknowledged" : true, "matchedCount" : 3, "modifiedCount" : 3 }
        ```
    *   **State After:** All documents in `products` with `category: "Electronics"` will now have an additional field `inStock: true`.

##### MongoDB Operators (used within `find`, `updateMany`, etc.)

Let's assume the `products` collection has the documents from above.

*   **$eq: { field: { $eq: value } }**
    *   **Example Input:** `> db.products.find({ price: { $eq: 75 } })`
    *   **Example Output:**
        ```json
        { "_id" : ObjectId("anotherGeneratedId2"), "name" : "Keyboard", "price" : 75, "category" : "Electronics", "inStock" : true }
        ```

*   **$ne: { field: { $ne: value } }**
    *   **Example Input:** `> db.products.find({ name: { $ne: "Laptop" } })`
    *   **Example Output:** (Mouse and Keyboard documents)

*   **$gt: { field: { $gt: value } }**
    *   **Example Input:** `> db.products.find({ price: { $gt: 100 } })`
    *   **Example Output:** (Laptop document)

*   **$gte: { field: { $gte: value } }**
    *   **Example Input:** `> db.products.find({ price: { $gte: 75 } })`
    *   **Example Output:** (Laptop and Keyboard documents)

*   **$lt: { field: { $lt: value } }**
    *   **Example Input:** `> db.products.find({ price: { $lt: 100 } })`
    *   **Example Output:** (Mouse and Keyboard documents)

*   **$lte: { field: { $lte: value } }**
    *   **Example Input:** `> db.products.find({ price: { $lte: 25 } })`
    *   **Example Output:** (Mouse document)

*   **$in: { field: { $in: [value1, value2] } }**
    *   **Example Input:** `> db.products.find({ name: { $in: ["Mouse", "Monitor"] } })`
    *   **Example Output:** (Mouse document, assuming "Monitor" doesn't exist)

*   **$and: { $and: [ { field1: value1 }, { field2: value2 } ] }**
    *   **Example Input:** `> db.products.find({ $and: [ { price: { $gt: 50 } }, { category: "Electronics" } ] })`
    *   **Example Output:** (Laptop and Keyboard documents)

*   **$or: { $or: [ { field1: value1 }, { field2: value2 } ] }**
    *   **Example Input:** `> db.products.find({ $or: [ { name: "Laptop" }, { price: { $lt: 30 } } ] })`
    *   **Example Output:** (Laptop and Mouse documents)

*   **$not: { field: { $not: { $gt: value } } }** (Note: $not is an operator that affects other operators)
    *   **Example Input:** `> db.products.find({ price: { $not: { $gt: 100 } } })` (i.e., price <= 100)
    *   **Example Output:** (Mouse and Keyboard documents)

---

#### Neo4j Cypher Query Language

**Initial State:** Assume an empty Neo4j database. Cypher queries are typically run in the Neo4j Browser or via a driver. The output shown is conceptual; actual Neo4j Browser output is tabular.

1.  **CREATE (n:Label {property1: value1, property2: value2})**
    *   **Command:** `CREATE (p:Person {name: "Alice", age: 30})`
    *   **Example Input:** `CREATE (m:Movie {title: "The Matrix", released: 1999})`
    *   **Example Output (Neo4j Browser):** `Added 1 label, created 1 node, set 2 properties.`
    *   **State After:** A node with label `Movie` and properties `title` and `released` is created.

2.  **MERGE (n:Label {property: value})**
    *   **Command:** `MERGE (c:City {name: "London"})`
    *   **Example Input (first time):** `MERGE (u:User {userId: "user123"})`
    *   **Example Output:** `Added 1 label, created 1 node, set 1 property.`
    *   **Example Input (second time, same query):** `MERGE (u:User {userId: "user123"})`
    *   **Example Output:** `(No changes)` or `Matched 1 node, set 0 properties.` (If it finds the node by `userId`)
    *   **State After:** Ensures a node `User` with `userId: "user123"` exists. Creates it if not found.

3.  **MATCH (a:Label1), (b:Label2) CREATE (a)-\[r:RELATIONSHIP_TYPE {property: value}]->(b)**
    *   **Command:** `MATCH (p:Person {name:"Alice"}), (m:Movie {title:"Inception"}) CREATE (p)-[r:WATCHED {rating: 5}]->(m)`
    *   **Example Input:** (Assuming Alice node and The Matrix node exist)
        `CREATE (a:Person {name: "AliceWonders"})`
        `CREATE (m:Movie {title: "The Matrix"})`
        `MATCH (p:Person {name: "AliceWonders"}), (mov:Movie {title: "The Matrix"}) CREATE (p)-[rel:REVIEWED {stars: 4}]->(mov)`
    *   **Example Output (for CREATE relationship):** `Created 1 relationship, set 1 property.`
    *   **State After:** A `REVIEWED` relationship from "AliceWonders" to "The Matrix" with property `stars: 4` is created.

4.  **MATCH (a:Label1), (b:Label2) MERGE (a)-\[:RELATIONSHIP_TYPE]->(b)**
    *   **Command:** `MATCH (p1:Person {name:"Alice"}), (p2:Person {name:"Bob"}) MERGE (p1)-[f:FRIENDS_WITH]->(p2)`
    *   **Example Input:** (Assuming AliceWonders and a new Bob node)
        `CREATE (b:Person {name: "BobBuilder"})`
        `MATCH (pA:Person {name: "AliceWonders"}), (pB:Person {name: "BobBuilder"}) MERGE (pA)-[:KNOWS]->(pB)`
    *   **Example Output (first time):** `Created 1 relationship.`
    *   **Example Output (second time):** `(No changes)`
    *   **State After:** Ensures a `KNOWS` relationship exists from AliceWonders to BobBuilder.

5.  **MATCH (n:Label {property1: value1}) RETURN n**
    *   **Command:** `MATCH (p:Person {name: "Alice"}) RETURN p`
    *   **Example Input:** `MATCH (m:Movie {title: "The Matrix"}) RETURN m`
    *   **Example Output (conceptual table):**
        ```
        +---------------------------------------------------+
        | m                                                 |
        +---------------------------------------------------+
        | (:Movie {title: "The Matrix", released: 1999}) |
        +---------------------------------------------------+
        ```

6.  **MATCH (a:Label1)-\[r:RELATIONSHIP_TYPE]->(b:Label2) RETURN a, r, b**
    *   **Command:** `MATCH (p:Person)-[w:WATCHED]->(m:Movie) WHERE p.name = "Alice" RETURN p, w, m`
    *   **Example Input:** `MATCH (p:Person {name: "AliceWonders"})-[rev:REVIEWED]->(mov:Movie) RETURN p, rev, mov`
    *   **Example Output (conceptual table):**
        ```
        +----------------------------------------------------+--------------------------+---------------------------------------------------+
        | p                                                  | rev                      | mov                                               |
        +----------------------------------------------------+--------------------------+---------------------------------------------------+
        | (:Person {name: "AliceWonders"})                 | [:REVIEWED {stars: 4}]   | (:Movie {title: "The Matrix", released: 1999}) |
        +----------------------------------------------------+--------------------------+---------------------------------------------------+
        ```

7.  **MATCH (n:Label {property1: value1}) SET n.property2 = value2**
    *   **Command:** `MATCH (p:Person {name: "Alice"}) SET p.age = 31`
    *   **Example Input:** `MATCH (m:Movie {title: "The Matrix"}) SET m.genre = "Sci-Fi"`
    *   **Example Output:** `Set 1 property.`
    *   **State After:** The "The Matrix" node now has a `genre` property.

8.  **MATCH (a:Label1)-\[r:RELATIONSHIP_TYPE]->(b:Label2) SET r.property = value**
    *   **Command:** `MATCH (p:Person {name:"Alice"})-[w:WATCHED]->(m:Movie {title:"Inception"}) SET w.rating = 4`
    *   **Example Input:** `MATCH (:Person {name: "AliceWonders"})-[rev:REVIEWED]->(:Movie {title: "The Matrix"}) SET rev.comments = "Loved it!"`
    *   **Example Output:** `Set 1 property.`
    *   **State After:** The `REVIEWED` relationship now has a `comments` property.

9.  **MATCH (n:Label) RETURN n.property**
    *   **Command:** `MATCH (p:Person {name: "Alice"}) RETURN p.age`
    *   **Example Input:** `MATCH (m:Movie {title: "The Matrix"}) RETURN m.released`
    *   **Example Output (conceptual table):**
        ```
        +--------------+
        | m.released   |
        +--------------+
        | 1999         |
        +--------------+
        ```

---

#### HBase

**Initial State:** Assume an HBase instance is running. Commands are run in the HBase shell.
Output messages can vary slightly.

1.  **create 'table_name', 'column_family1', 'column_family2'**
    *   **Command:** `create 'mytable', 'cf1', 'cf2'`
    *   **Example Input:** `create 'employees', 'personal_info', 'job_details'`
    *   **Example Output:** `0 row(s) in X.XXX seconds` (or similar, indicating table creation)
    *   **State After:** Table `employees` with column families `personal_info` and `job_details` is created.

2.  **disable 'table_name'**
    *   **Command:** `disable 'mytable'`
    *   **Example Input:** `disable 'employees'`
    *   **Example Output:** `0 row(s) in Y.YYY seconds`
    *   **State After:** Table `employees` is disabled. Operations like `alter` can now be performed.

3.  **enable 'table_name'**
    *   **Command:** `enable 'mytable'`
    *   **Example Input:** `enable 'employees'`
    *   **Example Output:** `0 row(s) in Z.ZZZ seconds`
    *   **State After:** Table `employees` is re-enabled.

4.  **alter 'table_name', 'column_family'** (This syntax typically adds a CF, or modifies an existing one with options)
    *   **Command:** `alter 'mytable', 'new_cf'`
    *   **Example Input (adding a new CF):** `alter 'employees', {NAME => 'contact_info'}`
    *   **Example Output:** `Updating all regions with new schema...` followed by `0 row(s) in W.WWW seconds` or `1/1 regions updated.`
    *   **State After:** Table `employees` now has an additional column family `contact_info`.

5.  **describe 'table_name'**
    *   **Command:** `describe 'mytable'`
    *   **Example Input:** `describe 'employees'`
    *   **Example Output:** (Something like this)
        ```
        Table employees is ENABLED
        employees
        COLUMN FAMILIES DESCRIPTION
        {NAME => 'contact_info', BLOOMFILTER => 'ROW', ...}
        {NAME => 'job_details', BLOOMFILTER => 'ROW', ...}
        {NAME => 'personal_info', BLOOMFILTER => 'ROW', ...}
        3 row(s) in V.VVV seconds
        ```

6.  **list**
    *   **Command:** `list`
    *   **Example Input:** `list`
    *   **Example Output:**
        ```
        TABLE
        employees
        another_table
        2 row(s) in U.UUU seconds
        ```

7.  **put 'table_name', 'row_key', 'column_family:column', 'value'**
    *   **Command:** `put 'mytable', 'row1', 'cf1:col1', 'val1'`
    *   **Example Input:** `put 'employees', 'emp001', 'personal_info:name', 'John Doe'`
    *   **Example Output:** (Usually no direct output, or `0 row(s) in T.TTT seconds`)
    *   **Example Input (more data for same row):** `put 'employees', 'emp001', 'job_details:title', 'Software Engineer'`
    *   **Example Output:** (Same as above)
    *   **State After:** Row `emp001` in table `employees` has `personal_info:name = 'John Doe'` and `job_details:title = 'Software Engineer'`.

8.  **put 'table_name', 'row_key', 'column_family:column', 'new_value'** (This is the same as insert, it overwrites)
    *   **Command:** `put 'mytable', 'row1', 'cf1:col1', 'updated_val1'`
    *   **Example Input:** `put 'employees', 'emp001', 'personal_info:name', 'Jonathan Doe'`
    *   **Example Output:** (Usually no direct output)
    *   **State After:** Row `emp001`'s `personal_info:name` is now `'Jonathan Doe'`.

9.  **get 'table_name', 'row_key'**
    *   **Command:** `get 'mytable', 'row1'`
    *   **Example Input:** `get 'employees', 'emp001'`
    *   **Example Output:**
        ```
        COLUMN                         CELL
         job_details:title             timestamp=..., value=Software Engineer
         personal_info:name            timestamp=..., value=Jonathan Doe
        2 row(s) in S.SSS seconds
        ```

10. **scan 'table_name'**
    *   **Command:** `scan 'mytable'`
    *   **Example Input:** `scan 'employees'`
    *   **Example Output:** (Assuming another employee `emp002` was added)
        ```
        ROW                            COLUMN+CELL
         emp001                        column=job_details:title, timestamp=..., value=Software Engineer
         emp001                        column=personal_info:name, timestamp=..., value=Jonathan Doe
         emp002                        column=personal_info:name, timestamp=..., value=Jane Smith
        2 row(s) in R.RRR seconds
        ```
        (Output will list all cells for all rows in the table)

11. **count 'table_name'**
    *   **Command:** `count 'mytable'`
    *   **Example Input:** `count 'employees'`
    *   **Example Output:** `2 row(s) in Q.QQQ seconds` (If there are 2 rows: emp001, emp002)

This covers all the commands with illustrative examples. Remember that specific output formatting and messages might differ slightly across environments.

---

### **Flashcards (Chapter 4 - Conceptual & Shell Commands)**

**Questions:**

**Conceptual NoSQL:**
1.  What is a primary difference in schema handling between RDBMS and NoSQL databases?
2.  What does "horizontal scalability" (scale-out) mean in the context of databases?
3.  Name two types of NoSQL databases and a typical use case for each.
4.  Briefly explain the purpose of "replication" in NoSQL distribution models.
5.  Briefly explain the purpose of "sharding" in NoSQL distribution models.
6.  How can a document database handle rows (entities) with varying attributes more easily than a traditional relational database?
7.  For a movie recommender, would a user's watch history (list of movies, ratings, timestamps) be better stored in a key-value store or a document database if you need to query individual parts of the history? Why?

**Redis Shell Commands:**
8.  What Redis command adds "item3" to the end of a list stored at key `mylist`?
9.  How do you get all key-value pairs from a Redis hash stored at `user:profile:123`?
10. Which Redis command checks if "banana" is a member of the set `fruitset`?

**MongoDB Shell Commands:**
11. Write the MongoDB command to find all documents in the `articles` collection where the `author` field is "Jane Doe".
12. Which MongoDB operator is used to find documents where a field's value is *less than or equal to* a given value?
13. How do you insert two documents, `{item: "A"}` and `{item: "B"}`, into the `inventory` collection in a single MongoDB command?

**Neo4j Cypher Commands:**
14. What is the Cypher command to find all nodes labeled `:User` and return their `name` property?
15. Write a Cypher `MERGE` command to ensure a node with label `:City` and property `name: "London"` exists.

**HBase Shell Commands:**
16. What is the HBase command to create a table `metrics` with a column family `data`?
17. How do you retrieve all columns for row key `sensor001` from the HBase table `readings`?

**Answers:**

**Conceptual NoSQL:**
1.  RDBMS have fixed, predefined schemas (schema-on-write), while NoSQL databases often have flexible, dynamic schemas (schema-on-read) or are schema-less.
2.  Adding more servers to a distributed system to handle increased load or data volume, rather than upgrading a single server's resources.
3.  *Key-Value:* Caching (fast lookups by key). *Document:* Content management, user profiles (flexible, complex objects). *Graph:* Social networks (modeling relationships). *Column-Family:* Time-series data, analytics (efficient column-wise access).
4.  To create copies of data on multiple nodes for high availability, fault tolerance, and to improve read performance by distributing read requests.
5.  To partition a large dataset across multiple nodes (shards), so each node manages only a subset of the data, improving write performance and storage capacity.
6.  In a document database, each document can have its own structure. New fields can be added or existing ones omitted from individual documents without affecting others in the same collection. RDBMS require all rows in a table to conform to the predefined table schema.
7.  Document database. It allows storing the complex watch history as a nested array/object within the user's document, and parts of this history (e.g., specific ratings) can be queried or updated more easily than if the entire history was an opaque value in a key-value store.

**Redis Shell Commands:**
8.  `RPUSH mylist "item3"`
9.  `HGETALL user:profile:123`
10. `SISMEMBER fruitset "banana"`

**MongoDB Shell Commands:**
11. `db.articles.find({ author: "Jane Doe" })`
12. `$lte` (e.g., `{ field: { $lte: value } }`)
13. `db.inventory.insertMany([{item: "A"}, {item: "B"}])`

**Neo4j Cypher Commands:**
14. `MATCH (u:User) RETURN u.name`
15. `MERGE (c:City {name: "London"})`

**HBase Shell Commands:**
16. `create 'metrics', 'data'`
17. `get 'readings', 'sensor001'`

---

## Chapter 5: Data Streaming (Conceptual Focus)

### **Revision Notes (Chapter 5 - Conceptual & Exam Style)**

**1. Core Concepts (Exam Relevance: Nov 2024 Q3a(i), Oct Exam Q3a)**

*   **Event Streaming:**
    *   **Definition:** Continuous flow of individual *events* that record state changes, actions, or occurrences.
    *   **Characteristics:**
        *   **Event-driven:** Processing is triggered by the arrival of new events.
        *   **Unbounded data:** Streams are continuous and potentially infinite.
        *   Often processed by specialized event stream processing frameworks (e.g., Kafka Streams, Flink, Spark Streaming).
        *   Focuses on individual records representing discrete happenings.
    *   **In Detail for Exam:** Explain that an event is an immutable fact about something that happened. Event streaming processes these facts as they occur, often in real-time or near real-time. It's not just raw data flow but data that signifies a specific occurrence (e.g., a click, a transaction, a sensor reading changing state). This allows for immediate reaction and analysis based on these occurrences.
*   **Replication (in NoSQL distribution models, also relevant to streaming systems for fault tolerance):**
    *   **Definition:** The process of creating and maintaining multiple copies of data (or stream partitions) on different nodes or brokers.
    *   **Purpose in Streaming (e.g., Kafka):**
        *   **Fault Tolerance:** If a broker (server) holding a partition of a topic fails, other brokers with replicas of that partition can take over, ensuring no data loss and continuous availability of the stream.
        *   **High Availability:** The stream remains accessible even if some nodes fail.
        *   **Durability:** Messages are not lost if a single broker fails.
    *   **How it works (e.g., Kafka):** For a topic partition, one broker acts as the "leader" and other brokers act as "followers" (replicas). Producers write to the leader, and the leader replicates the data to its followers. Consumers typically read from the leader. If the leader fails, one of the in-sync followers can be elected as the new leader.
    *   **In Detail for Exam:** Explain that replication is a fundamental concept for building resilient and fault-tolerant distributed systems, including streaming platforms like Kafka. It involves making redundant copies of data (e.g., topic partitions in Kafka) across multiple servers (brokers). If one server fails, the data is still available from its replicas on other servers, preventing data loss and service interruption. Specify leader-follower replication as a common model.

**2. Differentiating Batch vs. Stream Processing (Exam Relevance: Oct Exam Q3a(i))**

| Feature           | Batch Processing                                        | Stream Processing                                           |
| :---------------- | :------------------------------------------------------ | :---------------------------------------------------------- |
| **Data Scope**    | Processes a large, bounded dataset (all at once).       | Processes unbounded, continuous data as it arrives.         |
| **Data Size**     | Typically large volumes (GBs, TBs, PBs).                | Data arrives in small, continuous pieces (events/records).  |
| **Latency**       | High (minutes, hours, days). Results are not immediate. | Low (milliseconds, seconds, or near real-time).             |
| **Processing Mode**| Data collected first, then processed.                   | Data processed as it is received.                           |
| **Analysis Type** | Complex analysis, historical trends, reporting.         | Real-time analytics, alerts, immediate response.            |
| **State Mgmt**    | State usually recomputed each batch or not maintained.  | Often requires managing state over time (e.g., windowing).  |
| **Input Data Characteristics for Exam:** | Finite, static, known size, processed after collection. | Infinite, dynamic, unknown size, processed on arrival. |

**3. Applying Batch vs. Stream Processing (Exam Relevance: Oct Exam Q3a(ii))**

*   **Scenario: Social Media Platform for Researchers**
    *   **Batch Processing Example:**
        *   **Task:** Generating a weekly report of the most influential researchers based on the total number of likes and comments received on their posts over the past week.
        *   **How Applied:** Collect all post, like, and comment data for the entire week. At the end of the week, run a batch job (e.g., MapReduce or Spark batch) to process this collected data, aggregate the metrics, and generate the report.
    *   **Stream Processing Example:**
        *   **Task:** Providing a real-time notification to a researcher when their post receives a new "like" or "comment."
        *   **How Applied:** As "like" or "comment" events occur, they are ingested into a streaming system. A stream processing application immediately processes these events and triggers a notification to the relevant researcher.

**4. Kafka Implementation in a Scenario (Exam Relevance: Oct Exam Q3b)**

*   **Scenario: Social Media Platform for Researchers - Application for Kafka Streaming**
    *   **Application:** Real-time activity feed or notification system.
*   **Kafka Implementation:**
    *   **Two (2) Kafka Topics:**
        1.  `new_posts_topic`: For when researchers create new posts.
        2.  `interactions_topic`: For when users interact with posts (likes, comments).
    *   **Example of each topic’s message contents:**
        1.  `new_posts_topic` message:
            ```json
            {
              "post_id": "p123",
              "researcher_id": "user456",
              "timestamp": "2024-10-27T10:00:00Z",
              "post_content_summary": "Exciting new findings on quantum entanglement..." // or full content
            }
            ```
        2.  `interactions_topic` message (for a like):
            ```json
            {
              "interaction_id": "i789",
              "post_id": "p123",
              "interacting_user_id": "user789",
              "interaction_type": "like",
              "timestamp": "2024-10-27T10:05:00Z"
            }
            ```
            (for a comment):
            ```json
            {
              "interaction_id": "i790",
              "post_id": "p123",
              "commenting_user_id": "user111",
              "comment_text": "Great work!",
              "interaction_type": "comment",
              "timestamp": "2024-10-27T10:06:00Z"
            }
            ```
    *   **Specific users (consumers) who will subscribe to each topic:**
        1.  `new_posts_topic`:
            *   **Consumer 1: Feed Generation Service:** Subscribes to generate personalized activity feeds for other researchers (e.g., show posts from researchers they follow).
            *   **Consumer 2: Search Indexing Service:** Subscribes to index new posts for search functionality.
        2.  `interactions_topic`:
            *   **Consumer 1: Notification Service:** Subscribes to send real-time notifications to the post owner about new likes/comments.
            *   **Consumer 2: Analytics Service:** Subscribes to perform real-time analytics on post engagement (e.g., trending posts).

**5. Kafka Concepts (Broker, Producer, Consumer - Exam Relevance: Jan 2025 (BA) Q3b(iii))**

*   **Broker:**
    *   **Role:** A Kafka server that acts as a message intermediary. It stores the published messages (organized into topic partitions).
    *   **Functionality:** Manages partitions, handles read/write requests from producers/consumers, replicates partitions for fault tolerance, and forms part of a Kafka cluster.
    *   **Example:** In a social media platform, multiple brokers would store partitions of topics like `new_posts_topic` and `interactions_topic`, ensuring data is distributed and resilient.
*   **Producers:**
    *   **Role:** Client applications that publish (write) messages (records) to Kafka topics.
    *   **Functionality:** Decide which topic and partition to send a message to (or let Kafka decide based on key/round-robin). Serialize messages.
    *   **Example:** When a researcher submits a new post on the social media platform, the application backend acts as a producer, sending a message containing post details to the `new_posts_topic`.
*   **Consumers:**
    *   **Role:** Client applications that subscribe to one or more topics and process the messages published to those topics.
    *   **Functionality:** Read messages from topic partitions in an ordered manner (within a partition). Consumers are typically part of a "consumer group" to distribute processing load. Track offsets to know which messages have been processed.
    *   **Example:** A notification service for the social media platform acts as a consumer, subscribing to the `interactions_topic`. When a "like" message appears, it processes it and sends a notification to the post's author.

---

### **Flashcards (Chapter 4 & 5 - Conceptual)**

**Questions:**

**Chapter 4 Conceptual:**

1.  What are the three components of the BASE consistency model?
2.  Name one advantage of NoSQL's flexible schema compared to RDBMS's rigid schema.
3.  If you have a dataset with many interconnected entities and need to query these connections efficiently, which NoSQL database type is most suitable?
4.  In the context of NoSQL, what is an "aggregate"?
5.  What is the primary goal of sharding a NoSQL database?

**Chapter 5 Conceptual:**

6.  What is the fundamental difference in data handling between batch processing and stream processing?
7.  Give an example of an "event" in the context of event streaming for an e-commerce website.
8.  What is the main benefit of "replication" for Kafka topic partitions?
9.  In Kafka, what is the role of a "producer"?
10. What is the role of a "consumer group" in Kafka?
11. Which Kafka component is responsible for storing the actual message data?
12. How does "event-time processing" differ from "processing-time processing" in stream analytics?

**Answers:**

**Chapter 4 Conceptual:**

1.  Basically Available, Soft state, Eventually consistent.
2.  It allows for easier evolution of the data structure, accommodating new or varying attributes without requiring schema migrations for the entire database.
3.  Graph Database.
4.  A collection of related data objects that are treated as a single unit, especially for updates and consistency.
5.  To distribute data across multiple servers (shards) to improve scalability (both storage and write performance) and potentially fault tolerance.

**Chapter 5 Conceptual:**

6.  Batch processing operates on large, bounded datasets collected over time, while stream processing operates on unbounded, continuous data as it arrives.
7.  A customer adding an item to their cart, a customer completing a purchase, a product view, a search query.
8.  Fault tolerance and high availability (if a broker holding a leader partition fails, a replica can take over).
9.  To publish (write) messages (records) to Kafka topics.
10. To allow multiple consumer instances to work together to process messages from a topic, distributing the load and enabling parallel processing (each partition is consumed by one consumer within the group).
11. Broker (specifically, the log segments for topic partitions on the broker's disk).
12. Event-time processing analyzes data based on the timestamp embedded in the event itself (when the event actually occurred), while processing-time processing analyzes data based on when the system processes the event. Event-time is crucial for accurate analysis when there are delays or out-of-order data.

---

### **Mini-Test (Chapter 4 & 5 - Conceptual)**

**Part 1: Multiple Choice (1 point each)**

1.  Which of the following is a characteristic of NoSQL databases rather than traditional RDBMS?
    *   a) ACID transactions as primary consistency model.
    *   b) Fixed, predefined schema.
    *   c) Horizontal scalability.
    *   d) SQL as the primary query language.

2.  For an application requiring very fast lookups of user session data using a session ID, which NoSQL type is often a good fit?
    *   a) Graph Database
    *   b) Document Database
    *   c) Column-Family Database
    *   d) Key-Value Store

3.  Distributing different portions of a dataset across multiple servers to improve write throughput is known as:
    *   a) Replication
    *   b) Sharding
    *   c) Aggregation
    *   d) Caching

4.  Which term describes the processing of data as it arrives, often with very low latency?
    *   a) Batch Processing
    *   b) ETL Processing
    *   c) Stream Processing
    *   d) Offline Processing

5.  In Kafka, which component is responsible for writing messages to a topic?
    *   a) Broker
    *   b) Consumer
    *   c) Producer
    *   d) Zookeeper

**Part 2: True/False (1 point each)**

1.  All NoSQL databases guarantee strong consistency (ACID). (T/F)
2.  Replication in Kafka helps in distributing the storage of unique data segments across brokers. (T/F)
3.  Event streaming focuses on processing large, bounded datasets that have been collected over time. (T/F)
4.  A Kafka broker stores messages organized into topics, which can be further divided into partitions. (T/F)

**Part 3: Short Answer (3 points each)**

1.  Describe a scenario where a document database would be more advantageous than a relational database for storing product catalog information.
2.  Explain the roles of "leader" and "follower" replicas for a Kafka topic partition.
3.  Contrast "batch processing" and "stream processing" in terms of their input data characteristics.

---
**Answer Key (Mini-Test - Chapter 4 & 5 Conceptual)**

**Part 1: Multiple Choice**
1.  c) Horizontal scalability.
2.  d) Key-Value Store.
3.  b) Sharding.
4.  c) Stream Processing.
5.  c) Producer.

**Part 2: True/False**
1.  False (Many NoSQL databases prioritize availability and partition tolerance, often offering eventual consistency via BASE).
2.  False (Replication creates *copies* of data segments for fault tolerance; sharding distributes *unique* segments).
3.  False (Event streaming processes continuous, unbounded events as they occur).
4.  True.

**Part 3: Short Answer**
1.  *Example Answer:* For a product catalog where products have highly variable attributes (e.g., electronics have specs like screen size, RAM, while clothing has size, color, material), a document database is advantageous. Each product can be a document with its own specific fields, avoiding many NULL columns or complex EAV models that might be needed in a relational database.
2.  *Example Answer:* For a Kafka topic partition, the "leader" replica is the one that handles all produce (write) requests from producers and serves data to consumers. "Follower" replicas passively copy data from the leader. If the leader fails, one of the in-sync followers is elected as the new leader, ensuring data availability and fault tolerance.
3.  *Example Answer:*
    *   **Batch Processing Input:** Data is bounded, finite, and typically collected over a period before processing begins. The entire dataset is known before processing.
    *   **Stream Processing Input:** Data is unbounded, continuous, and arrives in real-time or near real-time. The total dataset size is unknown, and data must be processed as it comes.

This set of revision materials should give you a solid foundation for Chapters 4 and 5, aligning with the style of questions seen in your past papers. Remember to cross-reference with your lecture slides for any deeper conceptual details that might be specific to your course.




Okay, let's create the revision notes, flashcards, and mini-tests for **Chapter 6 (MapReduce)** and **Chapter 7 (Hadoop Security)**, focusing on the conceptual aspects and question styles observed in your past papers.

## Chapter 6: MapReduce (Conceptual & Exam Style)

### **Revision Notes (Chapter 6 - Conceptual & Exam Style)**

**1. Core Concepts (Exam Relevance: Nov 2024 Q1a, Jan 2025 (BA) Q1, Oct Exam Q1a)**

*   **MapReduce Framework:** A programming model and processing technique for distributed computing on large datasets (Big Data). Part of the Hadoop ecosystem.
*   **Core Idea:**
    *   **Map Phase:** Processes input data in parallel to produce intermediate key-value pairs.
    *   **Reduce Phase:** Aggregates or processes the intermediate key-value pairs (grouped by key) to produce the final output.
*   **Role of Record Reader (in Map phase - Oct Exam Q1a):**
    *   **Definition:** The Record Reader is responsible for reading data from an InputSplit (a chunk of the input file).
    *   **Function:** It parses the raw data from the InputSplit and converts it into key-value pairs suitable for the Mapper function. The type of key-value pair depends on the `InputFormat` used (e.g., for `TextInputFormat`, the key is often the byte offset of the line, and the value is the line content itself).
    *   It provides the actual input that the `map()` function will process, record by record.
*   **Role of Combiner (in Map phase / Advantage - Jan 2025 (BA) Q1b, Oct Exam Q1a):**
    *   **Definition:** An optional, localized Reducer that runs on the same node where a Map task has completed, *before* the Map output is sent over the network to the Reducers.
    *   **Function:** It takes the intermediate key-value pairs produced by a *single* Mapper and performs a preliminary aggregation (similar to the Reducer's function).
    *   **Advantage:** Its primary advantage is to **reduce the amount of data shuffled** across the network from Mappers to Reducers. This saves network bandwidth and reduces the load on the Reducers, often leading to significant performance improvements, especially for aggregations like sums or counts (e.g., word count).
    *   **Constraint:** The Combiner function must be associative and commutative, just like the Reducer function it mimics, because it operates on a subset of data and the order of processing partial results should not affect the final outcome.
    *   *Diagram for Advantage:* Show Mapper outputting many `(key, value)` pairs. Then show a Combiner box on the map side consolidating these into fewer `(key, aggregated_value)` pairs before they are sent to the Shuffle phase.

**2. MapReduce Job Stages (Exam Relevance: Nov 2024 Q1b, Jan 2025 (BA) Q1a)**

*   **Detailed Workflow (be prepared to draw/describe these for a given scenario):**
    1.  **Input:** The raw data to be processed (e.g., a text file, data from HDFS).
    2.  **Splitting:** The input data is divided by the `InputFormat` into fixed-size pieces called InputSplits. Each InputSplit is assigned to one Mapper.
    3.  **Mapping:**
        *   Each Mapper task takes an InputSplit.
        *   The Record Reader within the Mapper reads records (key-value pairs) from the InputSplit.
        *   The user-defined `map()` function is applied to each record, producing zero or more intermediate key-value pairs.
        *   (Optional) If a Combiner is specified, it processes the output of each Mapper locally.
    4.  **Shuffling (and Sorting):**
        *   The framework collects all intermediate key-value pairs from all Mappers.
        *   Data is partitioned by key (using a Partitioner, default is hash-based) so that all pairs with the same key are sent to the same Reducer.
        *   Within each partition destined for a Reducer, the data is sorted by key. This grouping by key is crucial for the Reduce phase.
    5.  **Reducing:**
        *   Each Reducer task receives all intermediate values associated with a unique key (or a set of unique keys it's responsible for).
        *   The user-defined `reduce()` function is called once for each unique key, taking the key and an iterator of its associated values as input.
        *   The `reduce()` function processes these values and produces zero or more final output key-value pairs.
    6.  **Output:**
        *   The final output key-value pairs from all Reducers are written to an output location (e.g., HDFS files) by the `OutputFormat`.

**3. Types of MapReduce Jobs (Exam Relevance: Nov 2024 Q1a)**

*   **Single Mapper Jobs:**
    *   **Explanation:** These jobs only consist of the Map phase. There is no Reducer, and therefore no Shuffle or Sort phase for Reducer input.
    *   **Purpose:** Primarily used for data transformation or filtering where no aggregation across the entire dataset based on keys is needed. Examples include converting data formats, cleaning data line-by-line, or selecting specific records based on some criteria applied to each record independently.
    *   **Flow:** Input -> Splitting -> Mapping -> Output.
    *   **Detailed Explanation:** Each input split is processed by a mapper, and the output of each mapper is written directly to the output files. Since there's no reducer, there's no need to group keys. This is efficient when the operation on each piece of data is independent of others.
*   **Multiple Mappers Reducer Job:**
    *   **Explanation:** This is the standard MapReduce job structure involving multiple parallel Mapper tasks followed by one or more Reducer tasks (often multiple reducers for parallelism in the reduce phase as well). It includes all phases: Splitting, Mapping, Shuffling, Sorting, and Reducing.
    *   **Purpose:** Used for tasks that require grouping and aggregation of data across the entire dataset, such as counting, summing, finding averages, or joining datasets based on keys.
    *   **Flow:** Input -> Splitting -> (Multiple) Mapping -> Shuffling & Sorting -> (Multiple) Reducing -> Output.
    *   **Detailed Explanation:** Multiple mappers process different splits of the input data in parallel. Their intermediate outputs, which are key-value pairs, are then shuffled and sorted. This means all values belonging to the same key are grouped together and sent to a specific reducer. Each reducer then processes all values for the keys assigned to it, performing the aggregation or computation. This is the most common and powerful form of MapReduce job. *Exam tip: the "multiple mappers" part emphasizes the parallelism of the map phase.*

**4. Diagrammatic Representation (Exam Relevance: Nov 2024 Q1b, Oct Exam Q1b, May/June 2024 Q1b)**

*   For tasks like "counting word frequency" (Nov 2024 Q1b) or "logistic companies per seller category" (Oct Exam Q1b), you need to:
    1.  **Show Input:** The raw text or tables.
    2.  **Splitting:** Conceptually show the input divided (e.g., lines of text as separate inputs to mappers).
    3.  **Mapping:** For each input split/record:
        *   Clearly define the `map(key_in, value_in) -> list_of(key_out, value_out)` logic.
        *   Show example intermediate key-value pairs.
        *   *Word Count Example:* `map(doc_id, line_text) -> for each word in line_text: emit (word, 1)`
        *   *Logistic Co. Example:* `map(seller_info_or_delivery_info) -> emit (category, logistic_company)` or `emit (category, 1)` if just counting unique companies per category first.
    4.  **Shuffling & Sorting:** Show the intermediate pairs being grouped by `key_out`.
        *   *Word Count Example:* `(word, [1,1,1,...])`
        *   *Logistic Co. Example:* `(category, [logistic_co1, logistic_co2, logistic_co1])`
    5.  **Reducing:** For each grouped key:
        *   Clearly define the `reduce(key_out, list_of_values_in) -> list_of(final_key, final_value)` logic.
        *   Show example final output.
        *   *Word Count Example:* `reduce(word, list_of_ones) -> emit (word, sum(list_of_ones))`
        *   *Logistic Co. Example:* `reduce(category, list_of_companies) -> emit (category, unique_set(list_of_companies))` or `emit (category, count(unique_set(list_of_companies)))`
    6.  **Output:** The final aggregated results.

---

### **Flashcards (Chapter 6 - Conceptual & Exam Style)**

**Questions:**

1.  What are the two primary phases in the MapReduce programming model?
2.  What is the main responsibility of the `Record Reader` in the Map phase?
3.  What is the primary advantage of using a `Combiner` in a MapReduce job?
4.  In which MapReduce phase are intermediate key-value pairs grouped by key and sent to specific Reducers?
5.  What is the typical output of a Mapper function?
6.  What is the input to a Reducer function?
7.  Describe a "Single Mapper Job" and its typical use case.
8.  For a word count task, what would a typical Mapper emit for the input line "hello world hello"?
9.  For a word count task, if a Reducer receives `("hello", [1, 1, 1])`, what would it typically emit?
10. What does the "Splitting" phase achieve in MapReduce?
11. Why must a Combiner function be associative and commutative?
12. What does the "Output" phase in a MapReduce job entail?

**Answers:**

1.  Map Phase and Reduce Phase.
2.  To read data from an InputSplit and convert it into key-value pairs for the Mapper function.
3.  To reduce the amount of data shuffled across the network from Mappers to Reducers, improving performance.
4.  Shuffling (and Sorting) phase.
5.  Zero or more intermediate key-value pairs.
6.  A key and an iterator (or list) of all intermediate values associated with that key.
7.  A job with only a Map phase (no Reducer, Shuffle, or Sort). Used for data transformation or filtering where no aggregation across keys is needed.
8.  `("hello", 1)`, `("world", 1)`, `("hello", 1)`.
9.  `("hello", 3)`.
10. It divides the input data into fixed-size pieces called InputSplits, each of which is processed by a separate Mapper.
11. Because it processes a subset of data locally on the map side, and the order of processing these partial results (by the combiner and then the reducer) should not affect the final aggregated outcome.
12. The final output key-value pairs from all Reducers are written to an output location (e.g., HDFS files) by the `OutputFormat`.

---

## Chapter 7: Hadoop Security (Conceptual & Exam Style)

### **Revision Notes (Chapter 7 - Conceptual & Exam Style)**

**1. Importance of Security & Best Practices (Exam Relevance: Jan 2025 (BA) Q3a, May/June 2024 Q2c)**

*   **Employee Training and Awareness:**
    *   **Statement Discussion:** Human error is a significant vulnerability. Employees are often the first line of defense or the weakest link.
    *   **Importance:**
        *   Reduces risks from social engineering (phishing), accidental data exposure, and misuse of privileges.
        *   Promotes a security-conscious culture.
        *   Ensures understanding of policies, procedures, and the importance of data protection.
        *   Helps in identifying and reporting potential security incidents.
    *   **Key Training Areas:** Strong password practices, identifying phishing attempts, safe data handling (especially sensitive data), understanding access control policies, incident reporting procedures.
*   **Other Best Practices (mentioned in slides, can be part of broader security discussion):**
    *   Regular Updates & Patching
    *   Secure Configuration & Hardening
    *   Strong Authentication (Kerberos) & Authorization (ACLs, RBAC)
    *   Data Encryption (at rest & in transit)
    *   Comprehensive Auditing
    *   Principle of Least Privilege
    *   Continuous Monitoring & Incident Response

**2. Kerberos Key Concepts (Exam Relevance: Oct Exam Q3c)**

*   **Kerberos:** Recommended protocol for strong authentication in Hadoop. Involves a trusted third party (KDC).
*   **Key Concepts (explain and give Hadoop context examples):**
    *   **Principal:**
        *   **Explanation:** A unique identity for any entity (user or service) that participates in the Kerberos authentication process. It's like a username in the Kerberos system.
        *   **Format:** Typically `primary/instance@REALM`.
        *   **Hadoop Context Example:**
            *   User principal: `hduser@YOUR_REALM.COM` (for a Hadoop user).
            *   Service principal: `hdfs/namenode.example.com@YOUR_REALM.COM` (for the HDFS NameNode service), `yarn/resourcemanager.example.com@YOUR_REALM.COM` (for YARN ResourceManager).
    *   **Authentication Server (AS):**
        *   **Explanation:** Part of the Key Distribution Center (KDC). It's the first point of contact for a client (user or service) wishing to authenticate.
        *   **Function:** Verifies the client's identity (usually via a password or keytab) and, if successful, issues a Ticket Granting Ticket (TGT).
        *   **Hadoop Context Example:** When a user (`hduser`) wants to access Hadoop services, their client program first contacts the AS. The AS validates `hduser`'s credentials and issues a TGT.
    *   **Ticket (specifically Service Ticket - ST):**
        *   **Explanation:** A credential issued by the Ticket Granting Service (TGS, also part of KDC) that allows a client to access a specific service. It's encrypted with the service's secret key.
        *   **Function:** The client presents the TGT to the TGS to request a Service Ticket for a particular service. The TGS issues the ST. The client then presents this ST to the actual service (e.g., NameNode). The service can decrypt the ST (as it knows its own key) to verify the client's identity and the ticket's authenticity.
        *   **Hadoop Context Example:** After obtaining a TGT, `hduser`'s client requests an ST from the TGS for the HDFS NameNode service (`hdfs/namenode.example.com`). `hduser` then presents this ST to the NameNode to perform HDFS operations. The ST proves to the NameNode that `hduser` has been authenticated by the KDC.
*   **Illustrating Kerberos Operation (Diagram - Nov 2024 Q4c):**
    *   Refer to the diagram on slide 18 or 19 of your Chapter 7 notes.
    *   Key components to show: Client (User), Authentication Server (AS), Ticket Granting Server (TGS) (both AS & TGS are part of KDC), and Application Server (Service like NameNode).
    *   Flow:
        1.  Client -> AS: Request TGT (sends user ID).
        2.  AS -> Client: Encrypted TGT (using user's password hash) + Session Key_TGS.
        3.  Client decrypts with password, gets TGT and Session Key_TGS.
        4.  Client -> TGS: Request Service Ticket (sends TGT, Authenticator, Service ID).
        5.  TGS -> Client: Encrypted Service Ticket (using service's key) + Session Key_Service.
        6.  Client -> Application Server: Service Ticket + Authenticator.
        7.  Application Server validates ST and Authenticator, grants access.

**3. Other Security Aspects (from May/June 2024 Q2c - if relevant to Ch 7 scope)**

*   **Anonymisation:** Techniques to remove or obscure personally identifiable information (PII) from data to protect privacy, while still allowing data to be used for analysis.
*   **Strong Passwords:** (Covered in detail in Chapter 7 notes). Practices for creating and managing passwords that are difficult to guess or crack.
*   **Access Control:** Mechanisms (like ACLs, RBAC) that define who can access what resources and what operations they can perform.

---

### **Flashcards (Chapter 6 & 7 - Conceptual)**

**Questions:**

**Chapter 6 - MapReduce Conceptual:**
1.  What is the primary purpose of the "Shuffle and Sort" phase in MapReduce?
2.  When would you typically use a "Single Mapper Job" in MapReduce?
3.  What is the role of the `OutputFormat` in a MapReduce job?
4.  Why is reducing network traffic a key benefit of using a Combiner?
5.  If a MapReduce job involves joining two large datasets based on a common key, which general type of MapReduce job structure would it be?

**Chapter 7 - Hadoop Security Conceptual:**
6.  What are the three key Kerberos concepts often discussed in the context of Hadoop security?
7.  In Kerberos, which server issues the initial Ticket Granting Ticket (TGT)?
8.  What does a Service Ticket (ST) allow a Kerberos client to do?
9.  Why is "employee training and awareness" considered a best practice in Hadoop security?
10. What is the "Principle of Least Privilege"?
11. What is "Defense in Depth" in security?
12. In the Kerberos workflow, what does the client use to decrypt the TGT received from the AS?

**Answers:**

**Chapter 6 - MapReduce Conceptual:**
1.  To group all intermediate key-value pairs from Mappers by key and sort them, ensuring all values for a single key are sent to the same Reducer for processing.
2.  For data transformation or filtering tasks that do not require aggregation across different keys, such as format conversion or line-by-line data cleaning.
3.  To write the final output key-value pairs from the Reducers to the specified output location (e.g., HDFS).
4.  Because the Combiner performs local aggregation on the map-side, reducing the volume of intermediate data that needs to be transferred over the network to the reducers.
5.  A Multiple Mappers Reducer Job (or more generally, a standard MapReduce job with map, shuffle, and reduce phases).

**Chapter 7 - Hadoop Security Conceptual:**
6.  Principal, Authentication Server (AS), and Ticket (specifically Service Ticket and TGT).
7.  The Authentication Server (AS), which is part of the Key Distribution Center (KDC).
8.  It allows the client to authenticate itself to a specific service (e.g., HDFS NameNode, YARN ResourceManager) and access its resources.
9.  Because human error is a major vulnerability; training helps users understand risks, follow policies (like strong passwords), and recognize threats like phishing, thus strengthening overall security.
10. Granting entities (users, processes) only the minimum permissions necessary to perform their intended functions for the minimum amount of time.
11. Employing multiple layers of security controls so that if one layer is breached, other layers still provide protection.
12. The client's secret key, which is typically derived from the user's password.

---

### **Mini-Test (Chapter 6 & 7 - Conceptual & Exam Style)**

**Part 1: Multiple Choice (1 point each)**

1.  In a MapReduce job for word counting, the Combiner's primary role is to:
    *   a) Split the input text into words.
    *   b) Sort the words alphabetically before sending to reducers.
    *   c) Perform a local sum of word counts on the mapper node.
    *   d) Write the final word counts to HDFS.

2.  Which MapReduce job type is most suitable for converting a large dataset from CSV to Parquet format without any aggregation?
    *   a) Multiple Mappers Reducer Job
    *   b) Single Mapper Job
    *   c) Single Mapper Combiner Reducer Job
    *   d) A job requiring only the Shuffle phase.

3.  In Kerberos, what does a "Principal" represent?
    *   a) The central Key Distribution Center.
    *   b) A security policy.
    *   c) A unique identity for a user or service.
    *   d) An encrypted message.

4.  "Defense in Depth" in Hadoop security refers to:
    *   a) Encrypting all data by default.
    *   b) Using only one very strong security mechanism.
    *   c) Implementing multiple layers of security controls.
    *   d) Regularly auditing user access logs.

5.  The Kerberos Authentication Server (AS) is responsible for issuing:
    *   a) Service Tickets (ST)
    *   b) User passwords
    *   c) Session keys for services
    *   d) Ticket Granting Tickets (TGT)

**Part 2: True/False (1 point each)**

1.  The output of a MapReduce Mapper function is always a single key-value pair. (T/F)
2.  In the Kerberos authentication process, the client directly sends its password to the service it wants to access. (T/F)
3.  Employee training is primarily focused on technical aspects of Hadoop security and not on user behavior. (T/F)
4.  The `reduce()` function in MapReduce is called once for every intermediate value. (T/F)

**Part 3: Short Answer (3 points each)**

1.  Briefly explain the "Shuffle and Sort" phase in a MapReduce job and why it is essential.
2.  Describe two key pieces of information a client needs to obtain from the Kerberos KDC (AS and TGS) to access a specific service.
3.  Explain the role of the "Record Reader" within the Map phase of a MapReduce program.

---
**Answer Key (Mini-Test - Chapter 6 & 7 Conceptual)**

**Part 1: Multiple Choice**
1.  c) Perform a local sum of word counts on the mapper node.
2.  b) Single Mapper Job.
3.  c) A unique identity for a user or service.
4.  c) Implementing multiple layers of security controls.
5.  d) Ticket Granting Tickets (TGT).

**Part 2: True/False**
1.  False (A Mapper can emit zero, one, or multiple key-value pairs).
2.  False (The client authenticates with the KDC using its password/key; it presents tickets, not its password, to services).
3.  False (Employee training covers user behavior, recognizing threats like phishing, and adhering to policies, in addition to any relevant technical aspects).
4.  False (The `reduce()` function is called once for every unique intermediate *key*, receiving an iterator of all values associated with that key).

**Part 3: Short Answer**
1.  *Example Answer:* The "Shuffle and Sort" phase collects all intermediate key-value pairs from the Mappers. It then partitions this data based on the keys (so all identical keys go to the same Reducer) and sorts the data within each partition by key. This is essential because it groups all values associated with a particular key together, allowing the Reducer to efficiently process all relevant values for that key in a single `reduce()` call.
2.  *Example Answer:*
    1.  **Ticket Granting Ticket (TGT):** Obtained from the Authentication Server (AS) after initial user authentication. The TGT is proof that the user has been authenticated by the KDC.
    2.  **Service Ticket (ST):** Obtained from the Ticket Granting Service (TGS) by presenting the TGT. The ST is specific to the service the client wants to access and is encrypted with the service's secret key.
3.  *Example Answer:* The Record Reader is a component within each Map task. Its role is to take an InputSplit (a chunk of the input data assigned to that Mapper) and parse it into individual records, which are then presented as key-value pairs to the user-defined `map()` function for processing. For example, with `TextInputFormat`, it reads lines from the file split, where the key might be the byte offset and the value is the line of text.

This covers the conceptual aspects for Chapters 6 and 7 based on your exam paper trends. Remember to integrate these with the practical command/code knowledge for a complete revision!
