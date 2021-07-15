def get_sent(df):
    sia = SentimentIntensityAnalyzer()
    neg_scores = []
    neu_scores = []
    pos_scores = []
    comp_scores = []
    abs_text = df['abstract']
    for text in abs_text:
        neg_scores.append(sia.polarity_scores(text)['neg'])
        neu_scores.append(sia.polarity_scores(text)['neu'])
        pos_scores.append(sia.polarity_scores(text)['pos'])
        comp_scores.append(sia.polarity_scores(text)['compound'])
    df['abs_sent_neg'] = neg_scores
    df['abs_sent_neu'] = neu_scores
    df['abs_sent_pos'] = pos_scores
    df['abs_sent_comp'] = comp_scores
    return df

def plot_sent(df):
    plt.figure()
    df['abs_sent_neg'].plot(title='neg scores', kind='kde')
    plt.show()
    df['abs_sent_neu'].plot(title='neu scores', kind='kde')
    print
    plt.show()
    df['abs_sent_pos'].plot(title='pos scores', kind='kde')
    plt.show()
    df['abs_sent_comp'].plot(title='comp scores', kind='kde')
    plt.show()

    print("Stats: neg")
    r =  df['abs_sent_neg'].max() - df['abs_sent_neg'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neg'])
    Q1 = np.percentile(df['abs_sent_neg'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_neg'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_neg'])
    Qmax = np.max(df['abs_sent_neg'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: neu")
    r =  df['abs_sent_neu'].max() - df['abs_sent_neu'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neu'])
    Q1 = np.percentile(df['abs_sent_neu'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_neu'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_neu'])
    Qmax = np.max(df['abs_sent_neu'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: pos")
    r =  df['abs_sent_pos'].max() - df['abs_sent_pos'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_pos'])
    Q1 = np.percentile(df['abs_sent_pos'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_pos'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_pos'])
    Qmax = np.max(df['abs_sent_pos'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: comp")
    r =  df['abs_sent_comp'].max() - df['abs_sent_comp'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neu'])
    Q1 = np.percentile(df['abs_sent_comp'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_comp'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_comp'])
    Qmax = np.max(df['abs_sent_comp'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    #for sent in df['abs_sent']:

def main():
    pass

if __name__ == "__main__":
    main()