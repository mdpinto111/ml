CREATE PROCEDURE [master_query]
	
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;


DECLARE @TableList TABLE (
	TableName NVARCHAR(128),
	GroupByName NVARCHAR(128),
	WhereName NVARCHAR(128),
	isSingle BIT
);
-- leading_column is the column to join with

INSERT INTO @TableList (TableName, GroupByName, WhereName, isSingle)
VALUES
	('SH_PIRTEI_NISHOM', 'PRT_MIS_TIK',NULL, 1) --,
--	('SH_SHUMA_MAST', 'MST_TIK', 'MST_SHNAT_MAS', 1),
--	('SH_SHUMOT', 'SHM_TIK', 'SHM_SHNAT_MAS', 1),
--	('SH_INT_SHUMA', 'INT_SHM_TIK','INT_SHM_SM',1),
--	('MM_TIK_TO_BIK', 'MIS_YESHUT','SHNAT_MAS',1),
--	('MM_HEVROT', 'MIS_HEVRA',NULL,1),
--	('SH_DOH_KASPIM', 'KSP_TIK','KSP_SHNAT_MAS',1),
--	('EM_MAKOR_DOH', 'DOH_MIS_OSEK','DOH_TKUFA',0),
--	('M14_TZAMA', 'M_MIS_YESHUT', 'M_YEAR', 0)



DECLARE @sql NVARCHAR(MAX);

DECLARE @table_name NVARCHAR(128);
DECLARE @group_by_column NVARCHAR(128);
DECLARE @where_by_column NVARCHAR(128);
DECLARE @is_single BIT;

SELECT @sql = STRING_AGG(CAST(
				CASE
				    WHEN 
						COLUMN_NAME NOT IN ('ISN', 'UPDATE_DATE', 'insert_date', 'source_system', 'ZMAN', 'RowID', 'TMPTIHA', 'T_PTIHA', 'T_KNISA') 
						AND COLUMN_NAME NOT LIKE '%_IDKUN'
						AND COLUMN_NAME NOT LIKE '%_IDCUN'
						AND COLUMN_NAME NOT LIKE '%_KLITA'
						AND COLUMN_NAME NOT LIKE '%_TAR'
						AND COLUMN_NAME NOT LIKE 'TAR_%'
						AND COLUMN_NAME NOT LIKE '%_FAX'
						AND COLUMN_NAME NOT LIKE '%_TELFON'
					THEN
						CASE 
							WHEN DATA_TYPE = 'bit' 
							THEN COLUMN_NAME + ' AS ' + COLUMN_NAME + '_onehot'
							WHEN DATA_TYPE = 'numeric' AND NUMERIC_PRECISION = 1 AND NUMERIC_SCALE = 0
							THEN COLUMN_NAME + ' AS ' + COLUMN_NAME + '_onehot'
							ELSE COLUMN_NAME
						END
					ELSE NULL
				END as NVARCHAR(MAX)), CHAR(13)+CHAR(9)+CHAR(9)+CHAR(9)+',')
			FROM INFORMATION_SCHEMA.COLUMNS 
			WHERE TABLE_NAME = 'EM_MASTMAM' 
			AND DATA_TYPE IN ('int', 'bigint', 'decimal', 'numeric', 'float', 'real')

DECLARE @final_sql NVARCHAR(MAX) = CHAR(9) + 'SELECT TOP 5000 * FROM (SELECT ' + @sql + ' from EM_MASTMAM) as t0 ';


DECLARE @final_agg_sql NVARCHAR(MAX) = ''
DECLARE @where_column_data_type NVARCHAR(50);
DECLARE @where_by_value NVARCHAR(128) = '2022';
DECLARE @leading_table_join NVARCHAR(128) = 't0.TIK';
DECLARE @counter INT = 1;

DECLARE table_cursor CURSOR FOR 
SELECT TableName, GroupByName, WhereName, isSingle FROM @TableList;

OPEN table_cursor;
FETCH NEXT FROM table_cursor 
INTO @table_name, @group_by_column, @where_by_column, @is_single;


WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = '';
	
	IF @is_single = 0
    BEGIN
		SELECT @sql = STRING_AGG(CAST(
			CASE 
				WHEN 
					COLUMN_NAME NOT IN (@group_by_column, 'ISN', 'UPDATE_DATE', 'insert_date', 'source_system', 'ZMAN', 'RowID', 'TMPTIHA', 'T_PTIHA', 'T_KNISA') 
						AND COLUMN_NAME NOT LIKE '%_IDKUN'
						AND COLUMN_NAME NOT LIKE '%_IDCUN'
						AND COLUMN_NAME NOT LIKE '%_KLITA'
						AND COLUMN_NAME NOT LIKE '%_TAR'
						AND COLUMN_NAME NOT LIKE 'TAR_%'
						AND COLUMN_NAME NOT LIKE '%_FAX'
						AND COLUMN_NAME NOT LIKE '%_TELFON'
				THEN 
					'AVG(' + COLUMN_NAME + ') AS ' + @table_name + '_avg_' + COLUMN_NAME + 
					', MIN(' + COLUMN_NAME + ') AS ' + @table_name + '_min_' + COLUMN_NAME + 
					', MAX(' + COLUMN_NAME + ') AS ' + @table_name + '_max_' + COLUMN_NAME + 
					', COUNT(' + COLUMN_NAME + ') AS ' + @table_name + '_count_' + COLUMN_NAME 
				ELSE NULL 
			END as NVARCHAR(MAX)), CHAR(13)+CHAR(9)+CHAR(9)+CHAR(9)+',')
		FROM INFORMATION_SCHEMA.COLUMNS 
		WHERE TABLE_NAME = @table_name 
		AND DATA_TYPE IN ('int', 'bigint', 'decimal', 'numeric', 'float', 'real')
		AND COLUMN_NAME <> @group_by_column
		AND (COLUMN_NAME <> @where_by_column OR @where_by_column IS NULL);


		SET @sql = CHAR(9) + 'SELECT '+ CHAR(13) + CHAR(9) + CHAR(9) + CHAR(9) + 
					@group_by_column + ', ' + CHAR(13) + CHAR(9)+CHAR(9)+CHAR(9) +@sql + CHAR(13)+CHAR(9)+CHAR(9)+' FROM ' + @table_name;

		
		IF @where_by_column IS NOT NULL AND LEN(@where_by_column) > 0
		BEGIN

			SELECT @where_column_data_type = DATA_TYPE 
			FROM INFORMATION_SCHEMA.COLUMNS 
			WHERE TABLE_NAME = @table_name 
			AND COLUMN_NAME = @where_by_column
			AND DATA_TYPE = 'numeric' 
			AND NUMERIC_PRECISION = 6
			AND NUMERIC_SCALE = 0;

		
			IF @where_column_data_type IS NOT NULL AND LEN(@where_column_data_type) > 0
			BEGIN
				SET @sql = @sql + ' WHERE ' + @where_by_column + '>=' + @where_by_value + '00 AND ' + @where_by_column + ' < ' +
							CAST((CAST(@where_by_value AS INT) + 1) AS NVARCHAR) + '00';
			END
			ELSE
			BEGIN
				SET @sql = @sql + ' WHERE ' + @where_by_column + '=' + @where_by_value;
			END
		END

		SET @sql = @sql + CHAR(13)+CHAR(9)+CHAR(9) + ' GROUP BY ' + @group_by_column;
		
		SET @sql = CHAR(13) + CHAR(9) +'LEFT JOIN (' + CHAR(13) + CHAR(9) + @sql +  CHAR(13)+CHAR(9) +') as t' + CAST(@counter AS NVARCHAR) +
			' ON t' + CAST(@counter AS NVARCHAR) + '.' + @group_by_column +
			'=' + @leading_table_join + CHAR(13);
		-- SELECT @sql;
		SET @final_agg_sql = @final_agg_sql + @sql
	END
	ELSE
	BEGIN
		SELECT @sql = STRING_AGG(CAST(
				CASE				
					WHEN 
						COLUMN_NAME NOT IN ('ISN', 'UPDATE_DATE', 'insert_date', 'source_system', 'ZMAN', 'RowID', 'TMPTIHA', 'T_PTIHA', 'T_KNISA') 
						AND COLUMN_NAME NOT LIKE '%_IDKUN'
						AND COLUMN_NAME NOT LIKE '%_IDCUN'
						AND COLUMN_NAME NOT LIKE '%_KLITA'
						AND COLUMN_NAME NOT LIKE '%_TAR'
						AND COLUMN_NAME NOT LIKE 'TAR_%'
						AND COLUMN_NAME NOT LIKE '%_FAX'
						AND COLUMN_NAME NOT LIKE '%_TELFON'
					THEN 
						CASE 
							WHEN 
								DATA_TYPE = 'bit'  OR
								(DATA_TYPE = 'numeric' AND NUMERIC_PRECISION = 1 AND NUMERIC_SCALE = 0)
							THEN COLUMN_NAME + ' AS ' + @table_name + '_' + COLUMN_NAME + '_onehot'
							ELSE COLUMN_NAME + ' AS ' + @table_name + '_' + COLUMN_NAME
						END
					ELSE NULL
				END as NVARCHAR(MAX)), CHAR(13)+CHAR(9)+CHAR(9)+CHAR(9)+',')
			FROM INFORMATION_SCHEMA.COLUMNS 
			WHERE TABLE_NAME = @table_name 
			AND DATA_TYPE IN ('int', 'bigint', 'decimal', 'numeric', 'float', 'real')
			AND COLUMN_NAME <> @group_by_column
			AND (COLUMN_NAME <> @where_by_column OR @where_by_column IS NULL);

		

		SET @sql = 'SELECT ' + @group_by_column + ',' +@sql + CHAR(13)+CHAR(9)+CHAR(9) + ' FROM ' + @table_name + ' '
		
		IF @where_by_column IS NOT NULL AND LEN(@where_by_column) > 0
		BEGIN
			SET @sql = @sql + ' WHERE ' + @where_by_column + '=' + @where_by_value;
		END
		
		SET @sql = CHAR(13) + CHAR(9)+ 'LEFT JOIN (' + @sql + ') t' + CAST(@counter AS NVARCHAR) +
		' ON ' + @leading_table_join + ' = t' + CAST(@counter AS NVARCHAR) + '.' + @group_by_column;

		SET @final_sql = @final_sql + @sql

	END

	SET @counter = @counter+1;
    FETCH NEXT FROM table_cursor INTO @table_name, @group_by_column, @where_by_column, @is_single;
END;



CLOSE table_cursor;
DEALLOCATE table_cursor;


DECLARE @proc_sql NVARCHAR(MAX);
DECLARE @procName NVARCHAR(128) = '[yaakov_temp_sp]';

DROP PROCEDURE IF EXISTS yaakov_temp_sp;


SET @proc_sql = 
'-- =============================================' + CHAR(13) +
'-- Author:		Yaakov Hatam' + CHAR(13) +
'-- Create date: 03-march-25' + CHAR(13) +
'-- Updated date: ' + CONVERT(varchar(30), GETDATE(), 120) + CHAR(13) +
'-- Description:	Prepare feature table, single row for every TIK' + CHAR(13) +
'-- EXEC [yaakov_temp_sp]'+ CHAR(13) +
'-- =============================================' + CHAR(13) +
'CREATE PROCEDURE ' + @procName + CHAR(13) + CHAR(10) +
'AS' + CHAR(13) + CHAR(10) +
'BEGIN' + CHAR(13) + CHAR(10) +
@final_sql + @final_agg_sql + CHAR(13) + CHAR(10) +
'END;';


EXEC(@proc_sql);

-- GO



-- init
DROP TABLE IF EXISTS yaakov_temp_table
DROP TABLE IF EXISTS #yaakov_temp_frs


-- Create a temporary table to store the results
CREATE TABLE #yaakov_temp_frs (
    is_hidden BIT NOT NULL,
    column_ordinal INT NOT NULL,
    name SYSNAME NULL,
    is_nullable BIT NOT NULL,
    system_type_id INT NOT NULL,
    system_type_name NVARCHAR(256) NULL,
    max_length SMALLINT NOT NULL,
    precision TINYINT NOT NULL,
    scale TINYINT NOT NULL,
    collation_name SYSNAME NULL,
    user_type_id INT NULL,
    user_type_database SYSNAME NULL,
    user_type_schema SYSNAME NULL,
    user_type_name SYSNAME NULL,
    assembly_qualified_type_name NVARCHAR(4000),
    xml_collection_id INT NULL,
    xml_collection_database SYSNAME NULL,
    xml_collection_schema SYSNAME NULL,
    xml_collection_name SYSNAME NULL,
    is_xml_document BIT NOT NULL,
    is_case_sensitive BIT NOT NULL,
    is_fixed_length_clr_type BIT NOT NULL,
    source_server SYSNAME NULL,
    source_database SYSNAME NULL,
    source_schema SYSNAME NULL,
    source_table SYSNAME NULL,
    source_column SYSNAME NULL,
    is_identity_column BIT NULL,
    is_part_of_unique_key BIT NULL,
    is_updateable BIT NULL,
    is_computed_column BIT NULL,
    is_sparse_column_set BIT NULL,
    ordinal_in_order_by_list SMALLINT NULL,
    order_by_list_length SMALLINT NULL,
    order_by_is_descending SMALLINT NULL,
    tds_type_id INT NOT NULL,
    tds_length INT NOT NULL,
    tds_collation_id INT NULL,
    tds_collation_sort_id TINYINT NULL
);

INSERT INTO #yaakov_temp_frs EXEC sys.sp_describe_first_result_set N'yaakov_temp_sp';

-- SELECT * FROM #yaakov_temp_frs;

DECLARE @ColumnSQL nvarchar(max) = 'CREATE TABLE yaakov_temp_table (';
DECLARE @ColumnName nvarchar(256);
DECLARE @IsNullable bit;
DECLARE @SystemTypeName nvarchar(256);


DECLARE cur CURSOR FOR 
SELECT name, is_nullable, system_type_name FROM #yaakov_temp_frs  


OPEN cur
FETCH NEXT FROM cur INTO @ColumnName, @IsNullable, @SystemTypeName

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @ColumnSQL = @ColumnSQL + @ColumnName + ' ' + @SystemTypeName

    -- Handle nullable property
    IF @IsNullable = 1
    BEGIN
        SET @ColumnSQL = @ColumnSQL + ' NULL'
    END
    ELSE
    BEGIN
        SET @ColumnSQL = @ColumnSQL + ' NOT NULL'
    END

    -- Add a comma for separation unless it's the last column
    SET @ColumnSQL = @ColumnSQL + ', '

    FETCH NEXT FROM cur INTO @ColumnName, @IsNullable, @SystemTypeName
END

CLOSE cur
DEALLOCATE cur

SET @ColumnSQL = @ColumnSQL + ')';

EXEC(@ColumnSQL)

INSERT INTO yaakov_temp_table EXEC yaakov_temp_sp;

-- Clean up
DROP TABLE #yaakov_temp_frs;


-- DELETE DUPS
WITH CTE AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY TIK ORDER BY TIK) AS rn
    FROM yaakov_temp_table
)
DELETE FROM CTE
WHERE rn > 1;

-- add primary key
ALTER TABLE yaakov_temp_table
ADD CONSTRAINT PK_yaakov_temp_table PRIMARY KEY (TIK);






-- Check if this have better performance
-- ALTER TABLE yaakov_temp_table REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE);




END



