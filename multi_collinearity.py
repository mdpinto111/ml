import pandas as pd


class MultiCollinearityEliminator:
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    def createCorrMatrix(self, include_target=False):
        if include_target == False:
            df_temp = self.df.drop([self.target], axis=1)
            corrMatrix = df_temp.corr(method="pearson", min_periods=30).abs()
        elif include_target == True:
            corrMatrix = self.df.corr(method="pearson", min_periods=30).abs()
        return corrMatrix

    def createCorrMatrixWithTarget(self):
        corrMatrix = self.createCorrMatrix(include_target=True)
        corrWithTarget = (
            pd.DataFrame(corrMatrix.loc[:, self.target])
            .drop([self.target], axis=0)
            .sort_values(by=self.target)
        )
        print(corrWithTarget, "\n")
        return corrWithTarget

    def createCorrelatedFeaturesList(self):
        corrMatrix = self.createCorrMatrix(include_target=False)
        colCorr = []
        for column in corrMatrix.columns:
            for idx, row in corrMatrix.iterrows():
                if (row[column] > self.threshold) and (row[column] < 1):
                    if idx not in colCorr:
                        colCorr.append(idx)
                    if column not in colCorr:
                        colCorr.append(column)
        print(colCorr, "\n")
        return colCorr

    def deleteFeatures(self, colCorr):
        corrWithTarget = self.createCorrMatrixWithTarget()
        for idx, row in corrWithTarget.iterrows():
            print(idx, "\n")
            if idx in colCorr:
                self.df = self.df.drop(idx, axis=1)
                break
        return self.df

    def autoEliminateMulticollinearity(self):
        colCorr = self.createCorrelatedFeaturesList()
        while colCorr != []:
            self.df = self.deleteFeatures(colCorr)
            colCorr = self.createCorrelatedFeaturesList()
        return self.df
