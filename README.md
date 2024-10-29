# 2OD
### Online Distributed Outlier Detection (2OD)

**Rebecca Salles, Benoit Lange, Reza Akbarinia, Eduardo Ogasawara, Esther Pacitti, Florent Masseglia**

- National Institute for Research in Digital Science and Technology (INRIA), Montpellier, France
- University of Montpellier, Montpellier, France
- Federal Center for Technological Education of Rio de Janeiro (CEFET/RJ), Rio de Janeiro, Brazil

We introduce the Online Distributed Outlier Detection (2OD) methodology, a novel well-defined, repeatable process for conducting scalable distributed multivariate anomaly detection. It is designed to expand the applicability of traditional anomaly detection methods across high-dimensional, high-frequency, high-throughput streaming data contexts. 2OD provides a comprehensive benchmarking framework that enables researchers to assess both the efficiency and accuracy of centralized, online, and distributed strategies across a broad array of outlier detection methods, thereby extending the applicability of distributed analysis to previously offline or centralized algorithms. We provide an implementation of 2OD with a Spark-like data partitioning distribution setup and based on the [PyOD](https://github.com/yzhao062/pyod) library for outlier detection.

It is used to conduct an experimental analysis of the efficiency-accuracy trade-off of several distributed online multivariate anomaly detection algorithms. We analyze the scenarios in which time series anomaly detection benefits from parallel and distributed computing to improve execution efficiency over high-throughput and/or high-dimensional streaming multivariate time series datasets that contain both synthetic and real-world anomalous data, covering up to hundreds of millions of observations. With a distributed setup, the algorithms were able to gain substantial computational efficiency, reaching on average tens and up to hundreds in speedup without compromising detection accuracy performances, even for high-dimensional data. These results indicate the potential for distributed online anomaly detection over multivariate time series and motivate future advancements in this area of study.
