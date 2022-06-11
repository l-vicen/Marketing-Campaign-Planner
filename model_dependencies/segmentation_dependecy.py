# Dependencies
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import _tree
from sklearn import tree
import plotly.express as px
import plotly.graph_objects as go
import graphviz

def segment_customer_using(data, target_columns, y_column_target):

    """This function bring all inputs for CART Algorithm together."""

    # Select data for modeling
    X = data[target_columns]
    # st.write(X)

    y = data[y_column_target].values
    # st.write(y)

    # Fit the model and display results
    X_train, X_test, y_train, y_test, clf, graph = fitting(X, y, 'gini', 'best', 
                                                        mdepth=3, 
                                                        clweight=None,
                                                        minleaf=1000)

    if (len(target_columns) == 2):
        Plot_3D(X, X_test, y_test, clf, x1=target_columns[0], x2=target_columns[1], mesh_size=1, margin=1)
    else: 
        st.warning('If you had picked 2 variables only, you would be able to see a 3D Plot here, but no worries!')

    #iterate_through_tree(clf)
    # get_lineage(clf, clf.tree_.feature)
    tree_to_code(clf, clf.tree_.feature)

def apply_weigthing_function(number_weights, data, column_names):

    weights = np.zeros(number_weights)

    for i in range(number_weights):
        weights[i] = st.number_input('Pick the weight for this attribute', key = column_names[i])
    
    #st.write('Weight Inputs {}'.format(weights))

    data_extended = data
    data_extended["Weight Score"] = 0.0

    for j in range(len(weights)):
        data_extended["Weight Score"] += weights[j] * data[column_names[j]]

    st.write(data_extended)

    # st.write('Weight Score {}'.format(weight_score))
    
    return data_extended

def fitting(X, y, criterion, splitter, mdepth, clweight, minleaf):

    """ Implementation according to https://towardsdatascience.com/cart-classification-and-regression-trees-for-clean-but-powerful-models-cc89e60b7a85,
    and adapted to our use case."""
    
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion, 
                                        splitter=splitter, 
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf, 
                                        random_state=0, 
                                  )
    clf = model.fit(X_train, y_train)

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    st.write('---')
    st.markdown('## Tree Summary')
    summary_dict = {
        'Classes: ': clf.classes_,
        'Tree Depth: ': clf.tree_.max_depth,
        'No. of leaves: ': clf.tree_.n_leaves,
        'No. of features: ': clf.n_features_in_,
    }

    st.table(pd.DataFrame.from_dict(summary_dict))

    st.write('---')
    st.markdown('## Evaluation on Test Data')

    score_te = model.score(X_test, y_test)

    text_one = 'Accuracy Score: {value}'
    st.info(text_one.format(value = score_te ))
    st.table(pd.DataFrame(classification_report(y_test, pred_labels_te, output_dict = True)).T)

    st.write('---')

    st.markdown('## Evaluation on Training Data')
    
    score_tr = model.score(X_train, y_train)
    text_two = 'Accuracy Score: {value}'
    st.info(text_two.format(value = score_tr))
    
    # Look at classification report to evaluate the model
    st.table(pd.DataFrame(classification_report(y_train, pred_labels_tr, output_dict = True)).T)
    
    st.markdown('---')
    st.markdown('## Regression Tree')
    # Use graphviz to plot the tree

    classes_for_plot = []

    for g in range(len(clf.classes_)):
        classes_for_plot.append(str(list(clf.classes_)[g]))

    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns, 
                                class_names=classes_for_plot,
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                               ) 

    graph = graphviz.Source(dot_data)


    st.graphviz_chart(dot_data, use_container_width=False)
    
    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf, graph


def Plot_3D(X, X_test, y_test, clf, x1, x2, mesh_size, margin):

    """ Visualization according to https://towardsdatascience.com/cart-classification-and-regression-trees-for-clean-but-powerful-models-cc89e60b7a85,
    and adapted to our use case."""
        
    # Specify a size of the mesh to be used
    mesh_size=mesh_size
    margin=margin

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
            
    # Calculate predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
    #fig = px.scatter_3d(x=[], y=[], z=[],
                     opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with CART Prediction Surface",
                      paper_bgcolor = 'white',
                      scene = dict(xaxis=dict(title=x1,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(title=x2,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(title='Probability (%)',
                                              backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0', 
                                              )))
    
    st.markdown('---')
    st.markdown('## 3D Plot')

    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='CART Prediction',
                              colorscale='Jet',
                              reversescale=True,
                              showscale=False, 
                              contours = {"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
    
    st.plotly_chart(fig, use_container_width=True)

def iterate_through_tree(clf):

    """
    Template from Package: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    and adapted to our case.
    """

    st.markdown('---')
   
    c1, c2 = st.columns((2, 1))
    
    c1.header('Tree Description')

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    c1.write(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )

    list_of_thresholds = []
    list_specific_threshold = []
    list_segments = []

    for i in range(n_nodes):

        if is_leaves[i]:

            c1.write(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )

            list_of_thresholds.append(list_specific_threshold)
            list_segments.append(i)

        else:
        
            list_specific_threshold.append(threshold[i]) 

            c1.write(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )

def tree_to_code(tree, feature_names):

    """Source from https://mljar.com/blog/extract-rules-decision-tree/ and adapted to our use case."""

    st.markdown('---')
    st.markdown('## Generated Code Snippet')
    st.info('The user can at this point copy and paste the code below into a python file. Next, he should put the correct location of his own CLV .csv file and run the code. The result is a dataframe with observations categorized according to the decision tree above.')

    snippet = "# Code Template" + "\n" +  "import pandas as pd" + "\n"+ "df = pd.read_csv(myCLV.csv)" + "\n" + "\n" + "...." + "\n"
              
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    first = "def predictor({}):".format(", ".join(str(f) for f in feature_names))
    # st.write(first)
    snippet = (snippet + "\n" + first + "\n")

    storage = []

    def recurse(node, depth, storage):
        
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # st.write("{}if {} <= {}:".format(indent, name, threshold))
            storage.append("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1, storage)

            # st.write("{}else:  # if {} > {}".format(indent, name, threshold))
            storage.append("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1, storage)

        else:  
            # st.write("{}return {}".format(indent, tree_.value[node]))
            storage.append("{}return {}".format(indent, tree_.value[node]))
            
    recurse(0, 1, storage)

    # st.title('Finale')
    # st.code(storage)

    for s in range(len(storage)):
        snippet = (snippet + storage[s] + "\n")

    st.code(snippet)
    return snippet
