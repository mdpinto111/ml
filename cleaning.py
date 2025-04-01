def remove_irrelevant_cols(df):
    df = df.drop(df.filter(like="T_IDCUN", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_IDKUN", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TIK", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_KLITA", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TAR", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="TAR_", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="ANAF", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_FAX", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="MIKUD", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_YESHUT", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TELFON", axis=1).columns, axis=1)

    drop_cols = [
        "DIRA",
        "EM_MAKOR_DOH_min_DOH_YESHUV_ESEK",
        "DIRA_ESEK",
        "TEL_ESEK",
        "KIDOMET_ESEK",
        "last_tik",
        "TA_DOAR",
        "TIK",
        "LSTNGR90_NO",
        "M90",
        "T_PTIHA",
        "TMPTIHA",
        "T_KNISA",
        "TIK_KODEM",
        "TA_DOAR",
        "YSHUV",
        "YSHUV_ESEK",
        "KIDOMET_ESEK",
        "DIRA_PRATI",
        "YSHUV_PRATI",
        "TEL_KODEM",
        "KOD_EMAIL",
        "SH_PIRTEI_NISHOM_PRT_TZ_BZ_1",
        "SH_PIRTEI_NISHOM_PRT_MIN_BZ_1_onehot",
        "SH_PIRTEI_NISHOM_PRT_MZV_MISH_BZR",
        "SH_PIRTEI_NISHOM_PRT_D_MZV_MISH",
        "SH_PIRTEI_NISHOM_PRT_TZ_BZ_2",
        "SH_PIRTEI_NISHOM_PRT_MIN_BZ_2_onehot",
        "SH_PIRTEI_NISHOM_PRT_D_BKST_SHN_ZUG",
        "SH_PIRTEI_NISHOM_PRT_D_SHDR_SHN_ZUG",
        "SH_PIRTEI_NISHOM_PRT_SHANA_AHARONA_DOCH",
        "SH_PIRTEI_NISHOM_PRT_OP_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_DIV_CLS_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_CLS_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_OP_D_ESK_NEW",
        "SH_PIRTEI_NISHOM_PRT_ZEHUT_MESHADER",
        "SH_PIRTEI_NISHOM_PRT_LAST_DATE_UPD",
        "SH_SHUMA_MAST_MST_BZ_MEVUTAL_onehot",
        "SH_SHUMA_MAST_MST_HAZHARA_ZUG_NIF_onehot",
        "SH_SHUMA_MAST_MST_MIS_MEYAZEG",
        "SH_SHUMA_MAST_MST_MIN1_onehot",
        "SH_SHUMA_MAST_MST_MIN2_onehot",
        "SH_SHUMA_MAST_MST_SUGTIK_PAIL_KODEM",
        "SH_SHUMA_MAST_MST_MOED_HUKI",
        "SH_SHUMA_MAST_MST_KOD_MISMAC",
        "SH_SHUMA_MAST_MST_KOD_NIHUL_S",
        "SH_SHUMA_MAST_MST_HACHNASA_HEV_BAIT",
        "SH_SHUMA_MAST_MST_MIS_YELADIM",
        "SH_SHUMA_MAST_MST_MAAM_103",
        "SH_SHUMA_MAST_MST_SEMEL_ORECH_SHUMA",
        "SH_SHUMA_MAST_MST_RAKAZ_ISHUR",
        "SH_SHUMA_MAST_MST_HANMAKA_KODEM",
        "SH_SHUMA_MAST_MST_YITRAT_SHANIM_LPRISA",
        "SH_SHUMOT_SHM_KOD_MAZAV",
        "SH_SHUMOT_SHM_ZEHUT_RASHUM",
        "SH_SHUMOT_SHM_MIN_RASHUM_onehot",
        "SH_SHUMOT_SHM_SEIF_SHUMA",
        "SH_SHUMOT_SHM_GERAON",
        "SH_SHUMOT_SHM_ZEHUT_MESHADER_NITUV_B",
        "SH_SHUMOT_SHM_ZEHUT_MEF_ISHUR",
        "SH_SHUMOT_SHM_SEMEL_MEASHER",
        "SH_SHUMOT_SHM_ZEHUT_BZ",
        "SH_SHUMOT_SHM_YITRAT_SHANIM_LPRISA",
        "SH_SHUMOT_SHM_SHIDUR_INT",
        "SH_SHUMOT_SHM_LLO_KOD_ISUF",
        "SH_INT_SHUMA_INT_SHM_Z_MESHADER",
        "SH_INT_SHUMA_INT_SHM_TIME_SHIDUR",
        "SH_INT_SHUMA_INT_SHM_BARCODE",
        "SH_INT_SHUMA_INT_SHM_ZEHUT_R",
        "SH_INT_SHUMA_INT_SHM_TEL_AVODA_R",
        "SH_INT_SHUMA_INT_SHM_ZEHUT_BZ",
        "SH_INT_SHUMA_INT_SHM_TEL_AVODA_BZ",
        "SH_INT_SHUMA_INT_SHM_TEL_BAIT",
        "SH_INT_SHUMA_INT_SHM_TEL_ACHER",
        "SH_INT_SHUMA_INT_SHM_MISPAR_OSEK",
        "SH_INT_SHUMA_INT_SHM_TEL_OZER",
        "SH_INT_SHUMA_INT_SHM_MIN_RASHUM",
        "SH_INT_SHUMA_INT_SHM_MIN_BZ",
        "MIS_HEVRA",
        "MM_HEVROT_MIS_REHOV",
        "MM_HEVROT_MIS_BAIT",
        "MM_HEVROT_SEMEL_ISHUV",
        "MM_HEVROT_SW_NIMRUR_KTVT_onehot",
        "MM_HEVROT_TA_DOAR",
        "MM_HEVROT_ISHUV_TA_DOAR",
        "MM_HEVROT_SUG_HEVRA_KODEM_onehot",
        "MM_HEVROT_KOD_MATARA",
        "MM_HEVROT_MIS_HEVRA_KODEM",
        "MM_HEVROT_STATUS_KODEM",
        "MM_HEVROT_TAT_SUG_onehot",
        "MM_HEVROT_TELEPHONE1",
        "MM_HEVROT_MIS_DIRA",
        "SH_DOH_KASPIM_KSP_ZEHUT",
        "SH_DOH_KASPIM_KSP_MIS_SHUTAFIM",
        "SH_DOH_KASPIM_KSP_MATBE_DIVUCH",
        "DOH_MIS_OSEK",
    ]

    return df.drop(drop_cols, axis=1)


def remove_null_cols(df):
    df_cleaned = df.dropna(axis=1, how="all")
    return df_cleaned


def remove_single_value_cols(df):
    return df.loc[:, df.nunique() != 1]
