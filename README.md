# Distributed Real-Time Inference pipeline using Ray in Python

Ray is an open-source distributed computing framework designed to simplify the development of scalable, high-performance AI workloads. 
At its core, Ray provides a general purpose, actor-based distributed runtime that allows Python programs to seamlessly
execute across a cluster of machines. More on Ray [here](https://www.ray.io).

Input is sourced from a *PostgreSQL database*, which holds personal and financial data
of users (from now, clients). The system must listen for insertions in a DB table on a
dedicated channel, retrieving and processing client records as soon as they are inserted
into the DB; thus, each record insertion is considered as an inference request.

The design of this database-driven pipeline introduces **three practical concerns**, which
inform the framework comparison: (1) it enables evaluating how different frameworks
manage database connections from distributed workers; (2) with client’sdata, it ispossible
to implement feature enrichment on-the-fly (computing a new client feature based on
existing features); and (3) it provides a natural way to implement **per-client state**.

In fact, there are multiple record insertions in the DB by the same client, and the state
tracks (that is, ***counts***) the number of inference requests received per client. This also
acts as a filtering mechanism: once a client exceeds a fixed threshold, their requests are
no longer processed. This stateful component also provides an opportunity to evaluate
how each framework supports stateful logic, both in terms of programming complexity
and its performance implications.

Valid requests then proceed to a preprocessing stage, where raw features are transformed
in the tensors the model expects. The inference step uses a **pre-trained Keras neural network**: each worker keeps a local copy of the model in memory, exploiting **data parallelism**
to avoid repeated I/O and reduce inference latency.

Finally, the pipeline concludes with **two database accesses**. First, the client’s record is
updated with the predicted value and enriched feature. Second, the system queries a separate financial product table, on the same DB, to retrieve personalized recommendations
based on the updated data.
