CREATE OR ALTER PROCEDURE usp_ValidateCreditCardDecisions
(
    @BatchSize INT = 10000
)
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Create temp table for batch processing
    CREATE TABLE #TempResults
    (
        RecordId INT,
        CreditCardContext VARCHAR(50),
        EnableContext INT,
        AccountStatusActive BIT,
        MonthEndBalance DECIMAL(18,2),
        LossAmount DECIMAL(18,2),
        LossDate DATE,
        ChargeOffDate DATE,
        ClosureAndStatusOffDate BIT,
        TotalRecoveries DECIMAL(18,2),
        ClosedChargedOffBalances DECIMAL(18,2),
        ChargeOffDateExtendedReporting DATE,
        ClosureDateExtendedReporting DATE,
        ValidationResult VARCHAR(20)
    );

    -- Create index for better performance
    CREATE CLUSTERED INDEX IX_Temp_RecordId ON #TempResults(RecordId);

    -- Table variable to store the starting point of each batch
    DECLARE @BatchStart INT = 0;

    WHILE EXISTS (
        SELECT 1 
        FROM CreditCardTransactions t
        WHERE t.RecordId > @BatchStart
    )
    BEGIN
        -- Clear temp results for new batch
        TRUNCATE TABLE #TempResults;

        -- Insert batch of records
        INSERT INTO #TempResults
        SELECT TOP (@BatchSize)
            t.RecordId,
            t.CreditCardContext,
            t.EnableContext,
            t.AccountStatusActive,
            t.MonthEndBalance,
            t.LossAmount,
            t.LossDate,
            t.ChargeOffDate,
            t.ClosureAndStatusOffDate,
            t.TotalRecoveries,
            t.ClosedChargedOffBalances,
            t.ChargeOffDateExtendedReporting,
            t.ClosureDateExtendedReporting,
            NULL as ValidationResult
        FROM CreditCardTransactions t
        WHERE t.RecordId > @BatchStart
        ORDER BY t.RecordId;

        -- Update last processed ID
        SELECT @BatchStart = MAX(RecordId) FROM #TempResults;

        -- Apply decision table rules
        UPDATE t
        SET ValidationResult = 
            CASE
                -- Rule 1: Base Cookie context with Enable Context 0 and Active Account
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 0 
                     AND AccountStatusActive = 1 
                THEN 'NOT/OPEN'

                -- Rule 2: Cookie context with Enable Context 2 and Loss Date
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND LossDate IS NOT NULL 
                     AND ChargeOffDate IS NULL 
                THEN 'YES/OPEN'

                -- Rule 3: Cookie context with Enable Context 2 and ChargeOff Date exists
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND ChargeOffDate IS NOT NULL 
                     AND LossDate IS NULL 
                THEN 'NOT/OPEN'

                -- Rule 4: Cookie context with both dates and closure status true
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND LossDate IS NOT NULL 
                     AND ChargeOffDate IS NOT NULL 
                     AND ClosureAndStatusOffDate = 1
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 5: Cookie context with both dates and closure status false
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND LossDate IS NOT NULL 
                     AND ChargeOffDate IS NULL 
                     AND ClosureAndStatusOffDate = 0
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 6: Negative balances
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND MonthEndBalance = -1 
                     AND LossAmount = -1 
                     AND LossDate IS NOT NULL
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 7: Charge off balances with extended reporting
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND ClosedChargedOffBalances IS NOT NULL 
                     AND ChargeOffDateExtendedReporting IS NOT NULL
                     AND TotalRecoveries > 0
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 8: Extended reporting with null charge off
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND ClosureDateExtendedReporting IS NOT NULL 
                     AND ChargeOffDate IS NULL
                     AND TotalRecoveries > 0
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 9: Extended reporting with closure status
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND ClosedChargedOffBalances IS NOT NULL 
                     AND ClosureDateExtendedReporting IS NOT NULL
                     AND ClosureAndStatusOffDate = 1
                THEN 'YES/CLOSED_CHARGED_OFF_CMT'

                -- Rule 10: Multiple statuses true
                WHEN CreditCardContext = 'Cookie' 
                     AND EnableContext = 2
                     AND LossDate IS NOT NULL 
                     AND ChargeOffDate IS NOT NULL 
                     AND ClosureAndStatusOffDate = 1
                     AND TotalRecoveries > 0
                THEN 'NOT/OPEN'

                -- Default fallback
                ELSE 'NOT/OPEN'
            END
        FROM #TempResults t;

        -- Update main transaction table with results
        UPDATE t
        SET ValidationResult = tmp.ValidationResult
        FROM CreditCardTransactions t
        INNER JOIN #TempResults tmp ON t.RecordId = tmp.RecordId;

        -- Progress logging
        DECLARE @ProcessedCount INT = @@ROWCOUNT;
        DECLARE @CurrentTime DATETIME = GETDATE();
        
        INSERT INTO ValidationProcessLog (BatchNumber, RecordsProcessed, ProcessDateTime)
        VALUES (@BatchStart, @ProcessedCount, @CurrentTime);

        RAISERROR ('Processed batch ending at ID %d with %d records at %s', 
                   0, 1, @BatchStart, @ProcessedCount, @CurrentTime) WITH NOWAIT;
    END

    -- Performance optimization: Update statistics
    UPDATE STATISTICS CreditCardTransactions WITH FULLSCAN;

    -- Cleanup
    DROP TABLE #TempResults;

    -- Final summary
    SELECT 
        ValidationResult,
        COUNT(*) as RecordCount,
        CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as DECIMAL(5,2)) as Percentage
    FROM CreditCardTransactions
    GROUP BY ValidationResult
    ORDER BY RecordCount DESC;
END;
GO

-- Create supporting log table if it doesn't exist
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'ValidationProcessLog') AND type in (N'U'))
BEGIN
    CREATE TABLE ValidationProcessLog (
        LogId INT IDENTITY(1,1) PRIMARY KEY,
        BatchNumber INT,
        RecordsProcessed INT,
        ProcessDateTime DATETIME,
        INDEX IX_ValidationProcessLog_DateTime (ProcessDateTime)
    );
END;
GO
