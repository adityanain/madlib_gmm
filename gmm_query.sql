/* 1. Prepare some input data. */

/* Create a table gmm_sample containing some sample data. */
CREATE TABLE public.gmm_sample(pid int, points double precision[]);

COPY gmm_sample (pid, points) FROM stdin DELIMITER '|';
1 | {14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.0600, 0.2800, 2.29, 5.64, 1.04, 3.92, 1065}
2 | {13.2, 1.78, 2.14, 11.2, 1, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.49, 1050}
3 | {13.16, 2.36,  2.67, 18.6, 101, 2.8,  3.24, 0.3, 2.81, 5.6799, 1.03, 3.17, 1185}
4 | {14.37, 1.95, 2.5, 16.8, 113, 3.85, 3.49, 0.24, 2.18, 7.8, 0.86, 3.45, 1480}
5 | {13.24, 2.59, 2.87, 21, 118, 2.8, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735}
6 | {14.2, 1.76, 2.45, 15.2, 112, 3.27, 3.39, 0.34, 1.97, 6.75, 1.05, 2.85, 1450}
7 | {14.39, 1.87, 2.45, 14.6, 96, 2.5, 2.52, 0.3, 1.98, 5.25, 1.02, 3.58, 1290}
8 | {14.06, 2.15, 2.61, 17.6, 121, 2.6, 2.51, 0.31, 1.25, 5.05, 1.06, 3.58, 1295}
9 | {14.83, 1.64, 2.17, 14, 97, 2.8, 2.98, 0.29, 1.98, 5.2, 1.08, 2.85, 1045}
10 | {13.86, 1.35, 2.27, 16, 98, 2.98, 3.15, 0.22, 1.8500, 7.2199, 1.01, 3.55, 1045}
\.

/* 
    2. Run kmeans algorithm on the sample data and create a temp table which 
      contains the data FROM gmm_sample and cluster label for each data point.
*/
CREATE TEMP TABLE gmm_sample_cluster AS 
    SELECT data.*, (madlib.closest_column(centroids, points)).column_id AS cluster_id
    FROM public.gmm_sample AS data,
        (  
            SELECT centroids
            FROM madlib.kmeanspp('gmm_sample', 
                                 'points', 
                                  2,
                                 'madlib.squared_dist_norm2',
                                 'madlib.avg', 
                                  20, 
                                  0.001
                                )
        ) AS centroids
    ORDER BY cluster_id;

/*
SELECT * FROM gmm_sample_cluster;
 pid |                               points                               | cluster_id 
-----+--------------------------------------------------------------------+------------
   1 | {14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065}  |          0
   2 | {13.2,1.78,2.14,11.2,1,2.65,2.76,0.26,1.28,4.38,1.05,3.49,1050}    |          0
   5 | {13.24,2.59,2.87,21,118,2.8,2.69,0.39,1.82,4.32,1.04,2.93,735}     |          0
   9 | {14.83,1.64,2.17,14,97,2.8,2.98,0.29,1.98,5.2,1.08,2.85,1045}      |          0
  10 | {13.86,1.35,2.27,16,98,2.98,3.15,0.22,1.85,7.2199,1.01,3.55,1045}  |          0
   6 | {14.2,1.76,2.45,15.2,112,3.27,3.39,0.34,1.97,6.75,1.05,2.85,1450}  |          1
   7 | {14.39,1.87,2.45,14.6,96,2.5,2.52,0.3,1.98,5.25,1.02,3.58,1290}    |          1
   3 | {13.16,2.36,2.67,18.6,101,2.8,3.24,0.3,2.81,5.6799,1.03,3.17,1185} |          1
   4 | {14.37,1.95,2.5,16.8,113,3.85,3.49,0.24,2.18,7.8,0.86,3.45,1480}   |          1
   8 | {14.06,2.15,2.61,17.6,121,2.6,2.51,0.31,1.25,5.05,1.06,3.58,1295}  |          1
*/


/*
SELECT cluster_id, madlib.array_avg(points, false) AS cluster_avg 
FROM gmm_sample_cluster
GROUP BY cluster_id
ORDER BY cluster_id;

 cluster_id |                                       cluster_avg                                       
------------+----------------------------------------------------------------------------------
          0 | {13.872,1.814,2.376,15.56,88.2,2.806,2.928,0.288,1.844,5.35198,1.044,3.348,988}
          1 | {14.036,2.018,2.536,16.56,108.6,3.004,3.03,0.298,2.038,6.10598,1.004,3.326,1340}
*/


/*
    Create an aggregate used to merge
    all the means in a single row.
*/
DROP AGGREGATE IF EXISTS array_agg_mult (anyarray);
CREATE AGGREGATE array_agg_mult (anyarray)  (
    SFUNC     = array_cat,
    STYPE     = anyarray,
    INITCOND  = '{}'
);


/*
    gmm_train requires mean matrix AS a 2-D array, so we 
    create create a 2-D matrix here FROM cluster_avg column
    in gmm_sample_cluster table.

    Input to array_to_2d is 1-D array in which the first
    two entries are the no. of rows and no. of columns 
    in the desired 2-D array.
*/
CREATE TEMP TABLE gmm_init_mean AS
    SELECT madlib.array_to_2d
        (
            ARRAY[c::double precision] || 
            ARRAY[(array_length(means_vec,1) / c)::double precision] || 
            means_vec 
        ) AS mean_matrix 
    FROM 
    ( 
        SELECT count(cluster_avg) AS c, array_agg_mult(cluster_avg) AS means_vec 
        FROM 
        (
            SELECT cluster_id, madlib.array_avg(points, false) AS cluster_avg
            FROM gmm_sample_cluster
            GROUP BY cluster_id
            ORDER BY cluster_id
        ) AS p
    ) AS q;

/*
                                mean_matrix                                                                             
-----------------------------------------------------------------------------------
 {{13.872,1.814,2.376,15.56,88.2,2.806,2.928,0.288,1.844,5.35198,1.044,3.348,988},
 {14.036,2.018,2.536,16.56,108.6,3.004,3.03,0.298,2.038,6.10598,1.004,3.326,1340}}

*/


/*
    Create a weight vector initialized with the 
    the value 1/k, where k is the number of clusters
*/
CREATE TEMP TABLE gmm_init_weight AS
    SELECT madlib.array_fill(
        madlib.array_of_float(array_length(points,1)), 
        (1.0/2)::double precision
    ) AS weight_vec
    FROM gmm_sample LIMIT 1;

/*
                      weight_vec                       
-------------------------------------------------------
 {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}

*/

/*
    Create a covariance matrix.
    The following queries creates an identity matrix.
*/
SELECT madlib.matrix_diag( 
    ( SELECT madlib.array_fill
            (
                madlib.array_of_float(array_length(points,1)), 
                1.0::double precision 
            ) FROM gmm_sample LIMIT 1
    ),
    'gmm_covar_temp'
);

SELECT madlib.matrix_densify('gmm_covar_temp', 'row="row", col="col", val=val', 'gmm_covar_temp_dense');

/* SELECT * FROM gmm_covar_temp_dense order by row; */

CREATE TEMP TABLE gmm_init_covar AS
    SELECT count(val) AS c, array_agg_mult(ARRAY[val]) AS covar_matrix 
    FROM (
        SELECT val FROM gmm_covar_temp_dense order by row
    ) AS covar_rows;

/*
    Create an table containing :
    weight vector in weight_vec column,
    mean matrix in mean_mat column,
    covariance matrix in covariance_matrix column.

    The naming of column is important as gmm_train expects
    the input in this format.
*/
DROP TABLE IF EXISTS gmm_init_model_table;
CREATE TABLE gmm_init_model_table AS
    SELECT
        gmm_init_weight.weight_vec AS weight_vector,
        gmm_init_mean.mean_matrix AS mean_matrix, 
        gmm_init_covar.covar_matrix AS covariance_matrix
    FROM gmm_init_weight, gmm_init_mean, gmm_init_covar;


