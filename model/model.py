import sys
import re
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from unidecode import unidecode
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import NMF
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial.distance import cosine
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment


def balance_classes(sm, X, y):
    """
    Args:
        sm (imblearn class): this is an imbalance learn oversampling or
        undersampling class
        X (2d numpy array): this is the feature matrix
        y (1d numpy array): this is the array of the targets
    Returns:
        X (2d numpy array): this is the balanced feature matrix
        y (1d numpy array): this is the corresponding balanced target array
    Returns X and y after being fit with the resampling method
    """
    X, y = sm.fit_sample(X, y)
    return X, y


def generate_dfs(file_idx):
    with gzip.open('data/listings.csv.gz', 'rb') as l:
        ls = pd.read_csv(l)
    with gzip.open('data/reviews.csv.gz', 'rb') as r:
        rv = pd.read_csv(r)
    for idx in xrange(file_idx):
        if idx > 0:
            l_name = 'data/listings ({}).csv.gz'.format(str(idx))
            r_name = 'data/reviews ({}).csv.gz'.format(str(idx))
            with gzip.open(l_name, 'rb') as lst:
                ls = pd.concat([ls, pd.read_csv(lst)], ignore_index=True)
            with gzip.open(r_name, 'rb') as rvs:
                rv = pd.concat([rv, pd.read_csv(rvs)], ignore_index=True)
            print(str(idx))
    return ls, rv


def write_to_csv(df_ls, df_rv):
    df_ls.to_csv('data/listings.csv')
    df_rv.to_csv('data/reviews.csv')


def drop_irrelevant_rows(df):
    '''
    this just drops the irrelevant rows
    '''
    df.drop(['access', 'availability_30', 'availability_365',
             'availability_60', 'availability_90', 'calendar_last_scraped',
             'calendar_updated', 'city', 'country', 'country_code',
             'experiences_offered', 'first_review', 'has_availability',
             'host_acceptance_rate', 'host_id', 'host_is_superhost',
             'host_location', 'host_listings_count', 'host_name',
             'host_neighbourhood', 'host_picture_url', 'host_response_rate',
             'host_response_time', 'host_since', 'host_thumbnail_url',
             'host_total_listings_count', 'host_url', 'house_rules',
             'interaction', 'jurisdiction_names', 'last_review',
             'last_scraped', 'last_scraped', 'latitude', 'license',
             'listing_url', 'longitude', 'market', 'medium_url',
             'neighbourhood', 'neighbourhood_cleansed',
             'neighbourhood_group_cleansed', 'notes', 'number_of_reviews',
             'picture_url', 'region_id', 'region_name', 'region_parent_id',
             'review_scores_accuracy', 'review_scores_checkin',
             'review_scores_cleanliness', 'review_scores_communication',
             'review_scores_location', 'review_scores_value',
             'reviews_per_month', 'scrape_id', 'smart_location', 'square_feet',
             'state', 'street', 'thumbnail_url', 'transit', 'xl_picture_url',
             'last_searched', 'host_verifications', 'zipcode',
             'monthly_price', 'calculated_host_listings_count'],
            axis=1, inplace=True)
    # df.drop(['Unnamed: 0', 'access', 'availability_30', 'availability_365',
    #          'availability_60', 'availability_90', 'calendar_last_scraped',
    #          'calendar_updated', 'city', 'country', 'country_code',
    #          'experiences_offered', 'first_review', 'has_availability',
    #          'host_acceptance_rate', 'host_id', 'host_is_superhost',
    #          'host_location', 'host_listings_count', 'host_name',
    #          'host_neighbourhood', 'host_picture_url', 'host_response_rate',
    #          'host_response_time', 'host_since', 'host_thumbnail_url',
    #          'host_total_listings_count', 'host_url', 'house_rules',
    #          'interaction', 'jurisdiction_names', 'last_review',
    #          'last_scraped', 'last_scraped', 'latitude', 'license',
    #          'listing_url', 'longitude', 'market', 'medium_url',
    #          'neighbourhood', 'neighbourhood_cleansed',
    #          'neighbourhood_group_cleansed', 'notes', 'number_of_reviews',
    #          'picture_url', 'region_id', 'region_name', 'region_parent_id',
    #          'review_scores_accuracy', 'review_scores_checkin',
    #          'review_scores_cleanliness', 'review_scores_communication',
    #          'review_scores_location', 'review_scores_value',
    #          'reviews_per_month', 'scrape_id', 'smart_location', 'square_feet',
    #          'state', 'street', 'thumbnail_url', 'transit', 'xl_picture_url',
    #          'last_searched', 'host_verifications', 'zipcode',
    #          'monthly_price', 'calculated_host_listings_count'],
    #         axis=1, inplace=True)
    return df


def featurize_text_information(df):
    '''

    '''
    description = df.description.values
    host_about = df.host_about.values
    name = df.name.values
    neighborhood_overview = df.neighborhood_overview.values
    space = df.space.values
    summary = df.summary.values
    review_text = df.review_text.values
    df.drop(['description', 'host_about', 'neighborhood_overview',
             'name', 'space', 'summary', 'review_text'],
            axis=1, inplace=True)
    return description, host_about, name, neighborhood_overview, space, summary, review_text


def convert_currency_to_float(number):
    if number != -999999:
        return float(number.replace('$', '').replace(',', ''))


def convert_target_info_categories(rating):
    if rating > 95.75:
        return '95+'
    elif rating > 90:
        return '90+'
    elif rating > 77.75:
        return '77+'
    else:
        return 'poor'


def convert_target_into_eight_categories(rating):
    if rating > 95:
        return 1
    # elif rating > 95.75:
    #     return 2
    elif rating > 90:
        return 2
    elif rating > 85:
        return 3
    elif rating > 77.75:
        return 4
    else:
        return 5


def binarize_target(rating):
    if rating > 90:
        return 1
    else:
        return 0


def simple_five_stars(rating):
    return int(round(float(rating)/20))


def grouped_five_star_rating(rating):
    computed_rating = int(round(float(rating)/20))
    if computed_rating == 5:
        return 3
    elif computed_rating == 4:
        return 2
    else:
        return 1


def load_cleaned_data():
    # df = pd.read_csv('data/listings.csv')
    df = pd.read_csv('data/dfls.csv')
    df['nan_rating'] = df.review_scores_rating.apply(np.isnan)
    df = df.query('nan_rating == False')
    df.drop('id', axis=1, inplace=True)
    df.drop('nan_rating', axis=1, inplace=True)
    df.host_has_profile_pic = df.host_has_profile_pic.apply(lambda x: 1 if x == 't' else 0)
    df.host_identity_verified = df.host_identity_verified.apply(lambda x: 1 if x == 't' else 0)
    df.instant_bookable = df.instant_bookable.apply(lambda x: 1 if x == 't' else 0)
    df.is_location_exact = df.is_location_exact.apply(lambda x: 1 if x == 't' else 0)
    df.require_guest_phone_verification = df.require_guest_phone_verification.apply(lambda x: 1 if x == 't' else 0)
    df.require_guest_profile_picture = df.require_guest_profile_picture.apply(lambda x: 1 if x == 't' else 0)
    df.requires_license = df.requires_license.apply(lambda x: 1 if x == 't' else 0)
    # r = pd.read_csv('data/reviews.csv')
    df = drop_irrelevant_rows(df)
    return df


def view_feature_importances(df, model):
    """
    Args:
        df (pandas dataframe): dataframe which has the original data
        model (sklearn model): this is the sklearn classification model that
        has already been fit (work with tree based models)
    Returns:
        nothing, this just prints the feature importances in descending order
    """
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))


def convert_amenity_string_into_set(amenity_string):
    if amenity_string == '{}':
        return ['no_amenities']
    else:
        amenity_string = amenity_string.replace('{', '')
        amenity_string = amenity_string.replace('}', '')
        amenity_string = amenity_string.split(',')
        return amenity_string


def create_master_set(all_amenities):
    master_amenities = set([item for row in all_amenities for item in row])
    return master_amenities


def create_amenities_columns(df, master_amenities):
    columns = list(master_amenities)
    for column in columns:
        df[column] = 0
    for column in columns:
        column_values = []
        for listing in df.amenities.values:
            if column in listing:
                column_values.append(1)
            else:
                column_values.append(0)
        df[column] = column_values
    return df


def create_master_dummies(df, column):
    new_dummy_columns = np.unique(df[column].values)
    for new_column in new_dummy_columns:
        df[new_column] = 0
    for new_column in new_dummy_columns:
        column_values = []
        for value in df[column].values:
            if value == new_column:
                column_values.append(1)
            else:
                column_values.append(0)
        df[new_column] = column_values
    return df


def feature_engineering(df):
    df.amenities = df.amenities.apply(convert_amenity_string_into_set)
    master_amenities = create_master_set(df.amenities.values)
    df = create_amenities_columns(df, master_amenities)
    df.drop('amenities', axis=1, inplace=True)
    df = df.fillna(-999999)
    df.cleaning_fee = df.cleaning_fee.apply(convert_currency_to_float)
    df.extra_people = df.extra_people.apply(convert_currency_to_float)
    df.price = df.price.apply(convert_currency_to_float)
    df.weekly_price = df.weekly_price.apply(convert_currency_to_float)
    df.security_deposit = df.security_deposit.apply(convert_currency_to_float)
    df.cleaning_fee = df.cleaning_fee.fillna(-999999)
    df.weekly_price = df.weekly_price.fillna(-999999)
    df.security_deposit = df.security_deposit.fillna(-999999)
    df.accommodates = df.accommodates.apply(float)
    df.bathrooms = df.bathrooms.apply(float)
    df.bedrooms = df.bedrooms.apply(float)
    df.beds = df.beds.apply(float)
    df.guests_included = df.guests_included.apply(float)
    df.minimum_nights = df.minimum_nights.apply(float)
    df['price_per_bed'] = df.price.values / (df.beds.values + 1)
    df['price_per_room'] = df.price.values / (df.bedrooms.values + 1)
    df['price_per_bathroom'] = df.price.values / (df.bathrooms.values + 1)
    df = create_master_dummies(df, 'bed_type')
    df = create_master_dummies(df, 'cancellation_policy')
    df = create_master_dummies(df, 'property_type')
    df = create_master_dummies(df, 'room_type')
    df.drop(['bed_type', 'cancellation_policy', 'property_type', 'room_type'], axis=1, inplace=True)
    return df


def count_vectorization(documents):
    documents = map(str, documents)
    documents = \
        [replace_number_in_sentence_with_num(document).translate(None, string.punctuation)
         for document in documents]
    vect = CountVectorizer(stop_words='english',
                           tokenizer=word_tokenize,
                           preprocessor=PorterStemmer().stem)
    tf_matrix = vect.fit_transform(documents)
    return vect, tf_matrix


def tfidf_vectorization(documents):
    documents = map(str, documents)
    documents = \
        [replace_number_in_sentence_with_num(document).translate(None, string.punctuation)
         for document in documents]
    vect = TfidfVectorizer(stop_words='english', max_features=8478,
                           tokenizer=word_tokenize,
                           preprocessor=PorterStemmer().stem)
    tf_matrix = vect.fit_transform(documents)
    return vect, tf_matrix


def replace_number_in_sentence_with_num(sentence):
    sentence = sentence.split()
    output = []
    for token in sentence:
        if re.search('\d+', token):
            output.append('num_')
        else:
            output.append(token)
    return ' '.join(output)


def fit_random_forest(X_train, y_train, X_test, y_test, df):
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, class_weight='balanced')
    print(np.mean(cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))
    print(confusion_matrix(y_test, rf.predict(X_test)))
    # view_feature_importances(df, rf)
    return rf, rf.predict_proba(X_train), rf.predict_proba(X_test)


def create_classifier_based_on_string_feature(string_feature_train, string_feature_test, y_train, y_test):
    mnb = MultinomialNB()
    vect, tf_matrix = count_vectorization(string_feature_train)
    print(np.mean(cross_val_score(mnb, tf_matrix, y_train, cv=10, n_jobs=-1, verbose=10)))
    string_feature_test = map(str, string_feature_test)
    string_feature_test = \
        [replace_number_in_sentence_with_num(document).translate(None, string.punctuation)
         for document in string_feature_test]
    tf_matrix_test = vect.transform(string_feature_test)
    mnb.fit(tf_matrix, y_train)
    print(mnb.score(tf_matrix_test, y_test))
    print(confusion_matrix(y_test, mnb.predict(tf_matrix_test)))
    return tf_matrix, tf_matrix_test, mnb.predict_proba(tf_matrix), mnb.predict_proba(tf_matrix_test)


def create_classifier_based_on_tfidf_feature(string_feature_train, string_feature_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200)
    vect, tf_matrix = tfidf_vectorization(string_feature_train)
    print(np.mean(cross_val_score(rf, tf_matrix, y_train, cv=10, n_jobs=-1, verbose=10)))
    string_feature_test = map(str, string_feature_test)
    string_feature_test = \
        [replace_number_in_sentence_with_num(document).translate(None, string.punctuation)
         for document in string_feature_test]
    tf_matrix_test = vect.transform(string_feature_test)
    rf.fit(tf_matrix, y_train)
    print(rf.score(tf_matrix_test, y_test))
    print(confusion_matrix(y_test, rf.predict(tf_matrix_test)))
    confusion(rf, tf_matrix_test, y_test)
    return tf_matrix, tf_matrix_test, rf.predict_proba(tf_matrix), rf.predict_proba(tf_matrix_test)


def gridsearch(paramgrid, model, X_train, y_train):
    """
    Args:
        paramgrid (dictionary): a dictionary of lists where the keys are the
        model's tunable parameters and the values are a list of the
        different parameter values to search over
        X_train (2d numpy array): this is the feature matrix
        y_train (1d numpy array): this is the array of targets
    Returns:
        best_model (sklearn classifier): a fit sklearn classifier with the
        best parameters from the gridsearch
        gridsearch (gridsearch object): the gridsearch object that has
        already been fit
    """
    gridsearch = GridSearchCV(model,
                              paramgrid,
                              n_jobs=-1,
                              verbose=10,
                              cv=10)
    gridsearch.fit(X_train, y_train)
    best_model = gridsearch.best_estimator_
    print('these are the parameters of the best model')
    print(best_model)
    print('\nthese is the best score')
    print(gridsearch.best_score_)
    return best_model, gridsearch


def reduce_dimensions(train_matrix, test_matrix):
    nmf = NMF(n_components=2)
    train_matrix = nmf.fit_transform(train_matrix)
    test_matrix = nmf.transform(test_matrix)
    return train_matrix, test_matrix


# def oversample(X_train, y_train):
#     # ros = RandomOverSampler
#     ros = RandomUnderSampler
#     X_train, y_train = ros.fit_sample(X_train, y_train)
#     return X_train, y_train

# review_train_tfidf, review_test_tfidf, review_y_pred_tfidf, review_y_pred_test_tfidf
# desc_train_tfidf, desc_test_tfidf, desc_y_pred_tfidf, desc_y_pred_test_tfidf
def get_sentiment_similarity(review_docs, description_docs):
    compound_difference = []
    neutral_difference = []
    positive_difference = []
    negative_difference = []
    for review, description in zip(review_docs, description_docs):
        review = str(review)
        description = str(description)
        rev_sid = vaderSentiment(review)
        desc_sid = vaderSentiment(description)
        compound_difference.append(desc_sid['compound']-rev_sid['compound'])
        neutral_difference.append(desc_sid['neu']-rev_sid['neu'])
        positive_difference.append(desc_sid['pos']-rev_sid['pos'])
        negative_difference.append(desc_sid['neg']-rev_sid['neg'])
    return compound_difference, neutral_difference, positive_difference, negative_difference

def mnb_scores(X_train, X_test, y_train, y_test):

    mnb=MultinomialNB()

    print(np.mean(cross_val_score(mnb, X_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    mnb.fit(X_train, y_train)
    print(mnb.score(X_test, y_test))
    print(confusion_matrix(y_test, mnb.predict(X_test)))
    confusion(mnb, X_test, y_test)

    return mnb, mnb.predict_proba(X_train), mnb.predict_proba(X_test)

def fit_SVC(X_train, X_test, y_train, y_test):
    clf=SVC()
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=5, verbose=5)))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(confusion_matrix(y_test, clf.predict(X_test)))
    confusion(clf, X_test, y_test)

    return clf, clf.predict_proba(X_train), clf.predict_proba(X_test)

#  Cost matrix to penalize the difference between misclassification of ordinal classes (rating levels: bad, good, best)
def cost():
    cost_mat=np.array([[0,10,20],[10,0,5],[10,5,0]])
    return cost_mat

#Normalize confusion matrix:
def normalize(cm):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def confusion(model, X_test, y_test):

    cost_mat=cost()
    mat = confusion_matrix(y_test, model.predict(X_test))
    upper_right=np.triu(mat,k=1)
    lower_left=np.tril(mat,k=-1)
    over_est = np.sum(upper_right)/float(np.sum(mat))
    under_est = np.sum(lower_left)/float(np.sum(mat))
    total_cost = np.sum(cost_mat*mat)
    print 'the confusion matrix is: '
    print (mat)
    print 'over-estimation rate: {}'.format(over_est)
    print 'under-estimation rate: {}'.format(under_est)
    print 'total cost of misclassification: {}'.format(total_cost)


if __name__ == '__main__':
    # reload(sys)
    # sys.setdefaultencoding("ISO-8859-1")
    df = load_cleaned_data()
    df = feature_engineering(df)
    train, test = train_test_split(df, test_size=.2)

    train_description, train_host_about, train_name, train_neighborhood_overview, train_space, train_summary, train_review_text = featurize_text_information(train)
    test_description, test_host_about, test_name, test_neighborhood_overview, test_space, test_summary, test_review_text = featurize_text_information(test)
    train.review_scores_rating = \
        train.review_scores_rating.apply(grouped_five_star_rating)
    test.review_scores_rating = \
        test.review_scores_rating.apply(grouped_five_star_rating)

    y_train = train.review_scores_rating.values
    y_test = test.review_scores_rating.values
    train.drop('review_scores_rating', axis=1, inplace=True)
    test.drop('review_scores_rating', axis=1, inplace=True)
    train.drop('Train', axis=1, inplace=True)
    test.drop('Train', axis=1, inplace=True)
    X_train = train.values
    X_test = test.values

    index = np.array(range(X_train.shape[0]))
    index_test = np.array(range(X_test.shape[0]))

    index, y_train = balance_classes(RandomUnderSampler(random_state=42), index.reshape(-1, 1), y_train)
    index_test, y_test = balance_classes(RandomUnderSampler(random_state=42), index_test.reshape(-1, 1), y_test)
    index = index.flatten()
    index_test = index_test.flatten()

    X_train = X_train[index]
    train_description = train_description[index]
    train_host_about = train_host_about[index]
    train_neighborhood_overview = train_neighborhood_overview[index]
    train_space = train_space[index]
    train_summary = train_summary[index]
    train_review_text = train_review_text[index]
    X_test = X_test[index_test]
    test_description = test_description[index_test]
    test_host_about = test_host_about[index_test]
    test_neighborhood_overview = test_neighborhood_overview[index_test]
    test_space = test_space[index_test]
    test_summary = test_summary[index_test]
    test_review_text = test_review_text[index_test]
    # # rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced')
    # # paramgrid = {'max_depth': [10, 20, 30],
    # #              'min_samples_leaf': [5, 7, 9],
    # #              'min_samples_split': [6, 8, 10]}
    # # best_model, gs = gridsearch(paramgrid, rf, X_train, y_train)
    print '---------------------------------------------------------------\n'
    print 'Random Forest Classifier on numerical and categorical features'
    rf, rf_y_pred, rf_y_pred_test = fit_random_forest(X_train, y_train, X_test, y_test, train)
    print '---------------------------------------------------------------\n'

    print 'Description string Random Forest'
    desc_train, desc_test, desc_y_pred, desc_y_pred_test = create_classifier_based_on_string_feature(train_description, test_description, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Host string Random Forest'
    h_a_train, h_a_test, h_a_y_pred, h_a_y_pred_test = create_classifier_based_on_string_feature(train_host_about, test_host_about, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Neighborhood string Random Forest'
    n_o_train, n_o_test, n_o_y_pred, n_o_y_pred_test = create_classifier_based_on_string_feature(train_neighborhood_overview, test_neighborhood_overview, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Space string Random Forest'
    spa_train, spa_test, space_y_pred, space_y_pred_test = create_classifier_based_on_string_feature(train_space, test_space, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Summary string Random Forest'
    sum_train, sum_test, summary_y_pred, summary_y_pred_test = create_classifier_based_on_string_feature(train_summary, test_summary, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Review string Random Forest'
    review_train, review_test, review_y_pred, review_y_pred_test = create_classifier_based_on_string_feature(train_review_text, test_review_text, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Random Forest on Review TFIDF matrix (unigram and bigram)'
    review_train_tfidf, review_test_tfidf, review_y_pred_tfidf, review_y_pred_test_tfidf = create_classifier_based_on_tfidf_feature(train_review_text, test_review_text, y_train, y_test)
    print '---------------------------------------------------------------\n'

    print 'Random Forest on Description TFIDF matrix (unigram and bigram)'
    desc_train_tfidf, desc_test_tfidf, desc_y_pred_tfidf, desc_y_pred_test_tfidf = create_classifier_based_on_tfidf_feature(train_description, test_description, y_train, y_test)
    print '---------------------------------------------------------------\n'

    desc_train, desc_test = reduce_dimensions(desc_train, desc_test)
    h_a_train, h_a_test = reduce_dimensions(h_a_train, h_a_test)
    n_o_train, n_o_test = reduce_dimensions(n_o_train, n_o_test)
    spa_train, spa_test = reduce_dimensions(spa_train, spa_test)
    sum_train, sum_test = reduce_dimensions(sum_train, sum_test)
    review_train, review_test = reduce_dimensions(review_train, review_test)

    review_train_tfidf, review_test_tfidf = reduce_dimensions(review_train_tfidf, review_test_tfidf)

    X_ensemble_train = \
        np.vstack((X_train.T,
                   desc_train.T,
                   h_a_train.T,
                   n_o_train.T,
                   spa_train.T,
                   sum_train.T,
                #    review_train.T,
                   review_train_tfidf.T,
                   rf_y_pred.T,
                   desc_y_pred.T,
                   h_a_y_pred.T,
                   n_o_y_pred.T,
                   space_y_pred.T,
                   summary_y_pred.T,
                #    review_y_pred.T,
                   review_y_pred_tfidf.T)).T
    X_ensemble_test = \
        np.vstack((X_test.T,
                   desc_test.T,
                   h_a_test.T,
                   n_o_test.T,
                   spa_test.T,
                   sum_test.T,
                #    review_test.T,
                   review_test_tfidf.T,
                   rf_y_pred_test.T,
                   desc_y_pred_test.T,
                   h_a_y_pred_test.T,
                   n_o_y_pred_test.T,
                   space_y_pred_test.T,
                   summary_y_pred_test.T,
                #    review_y_pred_test.T,
                   review_y_pred_test_tfidf.T)).T
    # columns = \
    #     list(train.columns)+['desc_1', 'desc_2', 'ha_1', 'ha_2',
    #                           'no_1', 'no_2', 'spa_1', 'spa_2',
    #                           'sum_1', 'sum_2']
    # columns += ['rf_pred_1', 'rf_pred_2', 'rf_pred_3', 'rf_pred_4', 'rf_pred_5', 'rf_pred_6']
    # columns += ['de_pred_1', 'de_pred_2', 'de_pred_3', 'de_pred_4', 'de_pred_5', 'de_pred_6']
    # columns += ['ha_pred_1', 'ha_pred_2', 'ha_pred_3', 'ha_pred_4', 'ha_pred_5', 'ha_pred_6']
    # columns += ['no_pred_1', 'no_pred_2', 'no_pred_3', 'no_pred_4', 'no_pred_5', 'no_pred_6']
    # columns += ['sp_pred_1', 'sp_pred_2', 'sp_pred_3', 'sp_pred_4', 'sp_pred_5', 'sp_pred_6']
    # columns += ['su_pred_1', 'su_pred_2', 'su_pred_3', 'su_pred_4', 'su_pred_5', 'su_pred_6']
    print 'Random Forest of Ensemble Monster: '
    rf_ensemble = RandomForestClassifier(n_jobs=-1, n_estimators=1000,
                                         max_depth=30,
                                         bootstrap=True, class_weight='balanced')
    # # # gb = GradientBoostingClassifier(n_estimators=500)
    # # paramgrid = {'max_depth': [10, 20, 30, 40],
    # #              'min_samples_leaf': [3, 5, 7, 9],
    # #              'min_samples_split': [4, 6, 8]}
    # # gridsearch(paramgrid, rf_ensemble, X_ensemble_train, y_train)
    # # print(np.mean(cross_val_score(rf_ensemble, X_ensemble_test, y_test, cv=5, n_jobs=-1, verbose=10)))
    # print(np.mean(cross_val_score(rf_ensemble, X_ensemble_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    rf_ensemble.fit(X_ensemble_train, y_train)
    print (np.mean(cross_val_score(rf_ensemble, X_ensemble_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    print(rf_ensemble.score(X_ensemble_test, y_test))
    print(confusion_matrix(y_test, rf_ensemble.predict(X_ensemble_test)))
    confusion(rf_ensemble, X_ensemble_test, y_test)
    print '---------------------------------------------------------------\n'
    # # # view_feature_importances(pd.DataFrame(X_ensemble_train, columns=columns), rf_ensemble)

    com, neu, pos, neg = get_sentiment_similarity(test_review_text,test_description)
