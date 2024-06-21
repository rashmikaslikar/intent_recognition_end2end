def test_dataset(df):
    """Test dataset quality and integrity."""
    column_list = ["date", "search_query", "market", "geo_country", "device_type", "browser_name", "intent", "daily_query_count"]
    df.expect_table_columns_to_match_ordered_list(column_list=column_list)  # schema adherence
    intents = ["TRANSLATION", "DICTIONARY", "NAVIGATION", "OTHER", "SHOPPING", "QA", "WEATHER", "COMPUTATION"]
    df.expect_column_values_to_be_in_set(column="intent", value_set=intents)  # expected labels
    df.expect_compound_columns_to_be_unique(column_list=["search_query"])  # data leaks
    df.expect_column_values_to_not_be_null(column="intent")  # missing values
    #df.expect_column_values_to_be_unique(column="id")  # unique values
    df.expect_column_values_to_be_of_type(column="search_query", type_="str")  # type adherence

    # Expectation suite
    expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
    results = df.validate(expectation_suite=expectation_suite, only_return_failures=True).to_json_dict()
    assert results["success"]