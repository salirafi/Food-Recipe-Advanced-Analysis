# content.py

NETWORK_CAPTION = """
Ingredient co-occurrence network across the whole recipes for the 100 most-used ingredients in the dataset. 
Each node/circles represents an ingredient (click "Labels Off" button on top-right to toggle the node labels on),
with node size proportional to the number of recipes containing it. Edges/lines connect
ingredients that frequently appear together, with width/thickness that depends on the number of the corresponding pairs occuring accross recipes. Node colour
encodes the log-scaled weighted degree. See "Insight" for explanation. The number of edges visualized are INTENTIONALLY reduced for faster rendering. 
The actual number that is used for analysis is much bigger.
"""

NETWORK_INSIGHT_TITLE = "Ingredient Co-occurrence Network"
NETWORK_INSIGHT_SUBTITLE = "How do ingredients connected to each other?"

NETWORK_INSIGHT_READ = """
<p>
Each circle or "node" represents an ingredient extracted from the recipe dataset, and the naming of each ingredient (most of them) have been standardized to make meaningful analysis.
The position of nodes is determined by the <a href=https://en.wikipedia.org/wiki/Force-directed_graph_drawing>Fruchterman-Reingold force-directed algorithm</a>, which places ingredients that frequently appear together closer in the graph.
As a result, <strong>ingredients that are commonly used in similar recipes tend to cluster spatially.</strong>
The node size tells the number of recipes containing the ingredient, which means that larger node tells that it is used in more recipes.
The node color shows the log-scaled weighted degree of the node, which defines the number of edges (see next) connected to the nodes. 
This means that darker nodes represent more edges connected to the corresponding nodes,
or in other words, these are ingredients that are more strongly connected to many other ingredients.
Finally, lines or "edges" connect ingredients that appear together in recipes, and the width of each edge corresponds to how frequent the two ingredients co-occur.
</p>
"""

NETWORK_INSIGHT_FINDINGS = """
<p>
From the 100 ingredients or nodes that plotted, they contribute to more than 4900 recipes. At the center lies a dense cluster of highly connected ingredients such as flour, sugar, butter, milk, eggs, salt, and vanilla. 
These ingredients act as structural "hubs" because they appear in many recipes and co-occur with a large number of other ingredients. 
Their strong connectivity results in darker colors (higher log weighted degree) and larger node sizes in the graph. 
Look at salt for example, if you hover its node, you will see that salt is connected to 99 other ingredients, which basically is the entire ingredients plotted!
That is why <strong>salt is ingredient/node with the highest degree in this plot</strong>, signifying its extreme importance among the world of foods.
If we moving outward, the network transitions from these common ingredients to less common ones, such as garlic, onion, parsley, 
and paprika, that connect different culinary contexts, and finally to specialized ingredients at the periphery that appear in fewer or more niche recipes.
If more ingredients were plotted, we would see around the periphery even less common recipes like brandy, asparagus, and even rum with smaller nodes and fewer edges than most.
Hence, this structure highlights how a relatively small set of foundational ingredients supports a much broader and diverse set of recipe combinations.
</p>
"""

NETWORK_INSIGHT_HEATMAP = """
<p>
The clustered heatmap below complements the network by showing the same co-occurrence structure but in matrix form.
To avoid messy visualization, only the 30 most-used ingredients are shown (zoom in for detailed figure). 
Each row and column is an ingredient, and darker cells indicate ingredient pairs that appear together more frequently across the recipes. 
Take salt-egg pair for example: its log value is around 11, which is one of the largest, meaning that this pair almost always exists in every recipe.
Because the rows and columns are hierarchically reordered, ingredients with similar co-occurrence patterns are placed next to one another. 
This helps confirm the dense forest seen in the network graph while also making it relatively easier to spot groups of ingredients that repeatedly travel together, 
even when those relationships are harder to isolate by eye in the network layout.
The white diagonal line consists of pairs with the same ingredient so we force-set the weight to zero because same-ingredient pairs, salt-salt for example, do not make sense.
</p>
"""

NETWORK_INSIGHT_TOPPAIR = """
<p>
For even clearer interpretation, the chart below complements the network and heatmap by isolating the strongest pairwise relationships, 
making it easier to identify which ingredient combinations form the most of the recipes in the dataset.
It highlights the 50 most frequent ingredient pairs in the network where each bar represents a pair of ingredients, 
ranked by the number of recipes in which they appear together. 
As expected, pairs involving highly ubiquitous ingredients such as salt, sugar, flour, and eggs dominate the ranking marking their foundational role across many types of recipes. 
<strong>So, make sure to at least have these ingredients, because with them, an entire world of recipes is already at your fingertips :)</strong>
</p>
"""

NETWORK_INSIGHT_METHOD = """
<p>
For each ingredient node, we first compute its weighted degree, defined as the total strength of its connections to other ingredients in the network. 
Each edge between two ingredients carries a weight equal to the number of recipes in which the pair co-occurs. 
Thus, the weighted degree of an ingredient is calculated by summing the weights of all edges connected to that node, 
representing the total frequency with which that ingredient appears together with other ingredients across the dataset. 
Because these values can vary widely (for example, some ingredients co-occurring thousands of times while others only by a few) the weighted degree is transformed using a logarithmic scaling, 
specifically \( \log(1 + k_w) \), where \(k_w\) is the weighted degree. 
This transformation compresses the range of values so that highly connected ingredients do not dominate the color scale,
 allowing all ingredients' nodes to remain visually distinguishable.
</p>
"""

NETWORK_HL_1_VALUE = "100"
NETWORK_HL_1_LABEL = "Nodes (ingredients)"
NETWORK_HL_2_VALUE = "4914"
NETWORK_HL_2_LABEL = "Edges (co-occur. pairs)"
NETWORK_HL_3_VALUE = "Salt"
NETWORK_HL_3_LABEL = "Highest-degree node"


LEIDEN_CAPTION = """
Ingredient community graph detected via the Leiden algorithm.
Each colour represents a distinct community of ingredients that tend to co-occur together.
All other features are the same as the co-occurrence graph. See "Insight" for explanation.
The number of edges visualized are INTENTIONALLY reduced for faster rendering. 
The actual number that is used for analysis is much bigger.
"""

LEIDEN_INSIGHT_TITLE = "Ingredient Community Graph Using Leiden Algorithm"
LEIDEN_INSIGHT_SUBTITLE = "Are there underlying patterns behind the connections?"

LEIDEN_INSIGHT_WHAT = """
<p>
The co-occurrence network already reveals which ingredients frequently appear together and highlights the most connected ingredients in the dataset. 
However, while that graph provides an overview of relationships, it can be difficult to clearly identify distinct ingredient groups simply by visual inspection since
the network is dense, with many overlapping connections, and clusters that appear visually may not be clearly separable. 
Therefore, <a href=https://en.wikipedia.org/wiki/Leiden_algorithm>the Leiden algorithm</a>, a kind of network clustering algorithm, is introduced to provide a systematic way to extract this hidden structure 
and to <strong>allow us from simply observing connections to formally identifying ingredient communities </strong>.
</p>
"""

LEIDEN_INSIGHT_READ = """
<p>
While all the other features such as nodes and edges define the same things as in the co-occurrence graph, here, the color of each node indicates its community membership, 
meaning that ingredients sharing the same color belong to the same community.
These communities are not manually-defined categories; instead, they emerge automatically from the structure of the ingredient co-occurrence network. 
A community should be interpreted as a region of the network where ingredients are used together more frequently than expected. 
However, this does not mean the community corresponds to a clean culinary category but instead reflects <strong>how ingredients behave in practice across many recipes</strong>.
This distinction is important since the graph is not trying to label cuisines, but to reveal the structure of ingredient usage.

"""

LEIDEN_INSIGHT_FINDINGS = """
<p>
At first glance, some communities do align with intuitive cooking patterns. 
For example, one of the most prominent clusters (dark purple) contains typical baking ingredients such as flour, sugar, eggs, vanilla, baking powder, and baking soda
which frequently appear together in cakes, cookies, and other desserts, forming a baking community. 
Another example is a different community (dark blue) that centers around, what seems to be, Mediterranean-style cooking ingredients such as olive oil, basil, oregano, tomatoes, and mozzarella, 
emphasizing herbs and fresh vegetable-based recipes.
However, the more interesting insight comes, not from the expected cuisine categories, but from where the clustering does not behave as expected.
For example, salt, which is one of the most universal ingredients, does not sit at the center of a large community but instead forms its own isolated cluster in community C7. 
Similarly, heavy cream somehow appears alone in community C8.
It may reveal something deeper about the data. 
Ingredients like salt appear across many different recipe types rather than within a single consistent group, hence, 
they connect broadly rather than densely within one region, leading the algorithm to not assign them to a dominant community.
Likewise, even within the “baking” cluster, there are unexpected members such as spices like clove, nutmeg, and cinnamon. 
This suggests that ingredient communities are not strict categories, but overlapping ecosystems where some ingredients bridge multiple cooking contexts.
</p>
<p>
The chart below provides a simple summary of the detected community structure by showing how many ingredients belong to each Leiden community. 
A few communities contain a large number of ingredients, while several others are much smaller and even consist of only one ingredient.
This uneven distribution reflects how ingredient ecosystems naturally form in the network where a small number of dominant culinary systems 
(such as baking or herb-based cooking) group many ingredients together, while unusual ingredients may end up forming 
very small communities (although that's not entirely true). Note that the number of ingredients that seems to be progressively going down from C0 to C8 is the artefact of community ordering within the code, not the natural structure of the network.
</p>
"""

LEIDEN_INSIGHT_INTERPRET = """
<p>
So what do we actually gain from this, if the clusters are imperfect?
</p>
<p>
The key value of this clustering is not in producing clean labels, but in revealing how structured (or unstructured) the ingredient space really is.
<strong>First</strong>, it confirms that strong ingredient ecosystems do exist. 
Baking ingredients, for example, form a dense and stable core, meaning these combinations are highly standardized across recipes.
<strong>Second</strong>, it highlights ingredients that behave differently from intuition. 
Salt being isolated tells us it is not part of any single culinary system but acts as a universal connector across many. 
Heavy cream forming its own cluster probably suggests that it appears in more specialized or context-specific combinations.
<strong>Third</strong>, it exposes the limitations of categorical thinking. 
Even within well-defined clusters, unexpected ingredients appear, showing that real-world cooking is more flexible than rigid categories like “dessert” or “savory”.
You can even put pineapple on top of pizza! Although...
</p>
<p>
Importantly, the results are sensitive to the resolution parameter δ (see <a href=https://leidenalg.readthedocs.io/en/stable/reference.html#rbconfigurationvertexpartition>here</a>; the plot uses 1.4). 
Lower values would merge communities into broader groups, while higher values would sets them further. 
This means the clustering should be interpreted as a "lens", not a ground truth.
</p>
"""

LEIDEN_INSIGHT_METHOD = """
<p>
The Leiden algorithm is a community detection method designed to identify groups of nodes in a network that are more densely connected internally than with the rest of the graph. 
It works iteratively through three main stages: moving nodes between communities to improve a quality score, refining the partition to ensure that communities remain internally well connected, 
and then aggregating each community into a new node to repeat the process on a simplified network.
The algorithm typically optimizes a measure called <i>modularity</i>, which evaluates how strong the community structure is compared to a random network. 
For this plot, modularity is defined as:
</p>

<p style="text-align:center;">
\( Q = Σ_{ij} (A_{ij} − \gamma ({k_i k_j \over 2m}) \delta(c_i, c_j) \)
</p>

<p>
Here, \( A_{ij} \) represents the connection strength between nodes \( i \) and \( j \), \( \delta \) is the resolution parameter, 
\( k_i \) and \( k_j \) are the total connection strengths of those nodes, 
m is the total edge weight of the network, and \( \delta (c_i, c_j) \) equals 1 when the two nodes belong to the same community. 
Maximizing this quantity means that the algorithm finds partitions where connections within communities are stronger than expected by chance.
</p>
"""


NUTRITION_CAPTION = """
A two-dimensional nutritional landscape of recipes constructed using Principal Component Analysis (PCA),
where each point represents a recipe positioned by its overall nutritional composition represented by the first and second principal component. 
Click "Labels Off" button to toggle on the axes' labels and ticks.
Colors indicate clusters identified using a Gaussian Mixture Model (GMM) to reveal natural groupings of recipes based on their nutritional structure.
See "Insight" for explanation.
"""
NUTRITION_INSIGHT_TITLE = "The Nutritional Landscape of Recipes"
NUTRITION_INSIGHT_SUBTITLE = "How recipes naturally organize by nutrition?"

NUTRITION_INSIGHT_WHAT = """
<p>
Before trying to understand the figure, it is important to understand that each recipe originally lives in a high-dimensional nutritional space that consists of
calories, fat, protein, etc. (as well as other derived quantities not originally present in the dataset).
Visualizing this directly is impossible, so to make this interpretable, we first use the so-called <a href=https://en.wikipedia.org/wiki/Principal_component_analysis>Principal Component Analysis</a> (PCA)
to <strong>compress all nutritional variables into two main axes that preserve as much variation as possible.</strong> 
In the figure, each point represents a recipe, and its position is determined by these two components: the first principal component PC1, which accounts for 33.2% of the dataset variance,
and the second principal component PC2, which accounts for 24.5% of the variance.
Together, they already summarize more than half of the nutritional diversity in the dataset. Recipes that are close together have similar nutritional profiles, 
even if they belong to very different categories.
Although PCA allows us to uncover the dominant directions of variation (essentially identifying the main “axes” along which recipes differ nutritionally),
alone it only reveals structure, not segmentation.
To help interpretation, we overlay a clustering model, specifically a <a href=https://www.ibm.com/think/topics/gaussian-mixture-model#:~:text=A%20Gaussian%20mixture%20model%20(GMM,weighted%20by%20a%20mixing%20coefficient.>Gaussian Mixture Model</a> (GMM),
which groups recipes into (seven) clusters. Unlike simpler methods, GMM assumes that each cluster follows a probabilistic distribution,
allowing soft boundaries and better handling of overlapping nutritional patterns.
This is why in the plot, clusters are not sharply separated; instead, they blend into each other.
</p>
"""

NUTRITION_INSIGHT_FINDINGS = """
<p>
Each point in the figure is a recipe, and the distance between points reflects similarity in nutritional composition. 
For example, the horizontal axis (PC1) separates low-calory vs high-calory recipes (among others),  
while the vertical axis (PC2) captures differences in sugar, fiber, and sodium content (again, among others).
Colors indicate cluster membership, with each cluster representing a group of recipes that share a similar nutritional structure. 
Several large-scale patterns become immediately visible. 
The landscape stretches roughly diagonally, indicating that the dominant variation is not isolated to a single nutrient but reflects combined trade-offs between the nutrients.
Recipes toward the right tend to be more nutricious (in absolute terms), while those on the left are more associated with "light snacks" recipes with low nutrition content.  
For example, on the right-hand side, clusters (e.g., clusters 2 and 6) concentrate recipes with higher protein, 
while the left-hand side, including clusters 3 and 5, captures recipes that are lower in protein. 
The lower region of the plot contains clusters characterized by weaker nutritional balance with low fiber and higher carbohydrates (not shown on the hover box),
while the upper recipes contain much more sugar than the lower ones, but both sides contain relatively similar sodium.
"""

NUTRITION_INSIGHT_LOADING = """
To understand the PCA plot further, the below "loading" plot is given.
The plot explains <strong>what drives the structure of the nutritional landscape</strong> by showing how each nutritional feature contributes to the two principal components.
Each bar represents the weight (loading) of a feature in defining either PC1 or PC2. 
a large positive loading means the feature pushes recipes toward the positive end of that axis, and a negative loading pulls them toward the negative end, 
while values near zero indicate little contribution.

PC1 is dominated by large positive loadings across nearly every feature such as 
Calories (~0.40), ProteinContent (~0.41), FatContent (~0.39), and CarbohydrateContent (~0.20), 
but with the notable exception of SugarPer100Cal (~−0.19) and SatFatShareOfFat (~−0.18), which are slightly negative. 
This tells us that PC1 is essentially an overall nutritional magnitude or "richness" axis 
where <strong>recipes scoring high on PC1 tend to be calorie-dense and macro-heavy across the board (high protein, fat, sodium, carbs), 
while recipes at the low end are lean and sparse.</strong> 
Because the per-100-calorie ratio features (ProteinPer100Cal, SugarPer100Cal) load differently from their raw counterparts, 
PC1 is specifically capturing absolute quantity rather than nutritional quality per calorie. 

PC2 tells a more nuanced story: it carries strong positive loadings on SugarContent (~0.50), CarbohydrateContent (~0.37), 
Calories (~0.25), FatContent (~0.33), and FiberContent (~0.16), 
but distinctly negative loadings on SugarShareOfCarbs (~−0.33), ProteinPer100Cal (~−0.19), SodiumContent (~−0.06), and SodiumPer100Cal (~−0.06). 
This means PC2 represents a sweet-versus-savory or carbohydrate-quality axis where 
<strong> a high PC2 score indicates a recipe is calorie-rich from carbohydrates and fat</strong>
but those carbohydrates come from complex or diverse sources (since SugarShareOfCarbs is negative, meaning sugar is a small fraction of the total carbs), 
<strong>while a low PC2 score would indicate high-protein, high-sodium, and sugar-dominant compositions</strong> 
typical of savory or processed foods. 
Together, these loadings show that the entire landscape is structured by two fundamental forces:
<strong>(1) total nutrient density and (2) the relative balance between sugar, protein, and other nutrients,</strong> 
providing the physical interpretation behind the PCA map and allowing us to understand why clusters and gradients appear in specific regions of the nutritional space.
</p>
"""

NUTRITION_INSIGHT_INTERPRET = """
<p>
<strong>This landscape reveals that recipes are not randomly distributed but organized along a few fundamental nutritional gradients.</strong> 
Rather than forming distinct and isolated groups, recipes occupy a continuous space shaped by trade-offs. 
The clustering helps summarize this structure, but it should not be interpreted as defining strict categories. 
One important caveat is that clusters are not compact islands but elongated regions, which reflects a key limitation of clustering: 
the boundaries are model-imposed rather than naturally existing. 
GMM improves flexibility compared to simpler clustering algorithm like K-means, 
but it still assumes Gaussian shapes, which may not perfectly match real nutritional distributions. 
As a result, some recipes near cluster boundaries could reasonably belong to multiple groups, which is partially captured by the model’s confidence scores,
and not entirely surprising because it is consistent with how real-world food behaves. 
To show this, when we overlay traditional recipe categories like shown below, we observe significant overlap, 
suggesting that <strong> common labels such as “Dessert” or “Meal” are not distinctly different and do not fully capture nutritional reality.</strong>
This means that, when your friend says they eat veggies everyday, depending on what veggies they eat, that does not necessarily mean that they are healthy,
because veggies spans a whole range of nutrient compositions.
</p>
"""
NUTRITION_INSIGHT_METHOD = """
<p>
We begin by deriving a set of nutritional features, including both absolute values, like calories and fat content, 
and normalized parameters, like protein per 100 calories. 
Skewed variables are log-transformed, and all features are standardized to ensure comparability. 
PCA is then applied to reduce dimensionality, extracting the two principal components that capture the largest variance in the dataset. 
While PC1 and PC2 already explain a substantial portion of the variation, higher components (PC3, PC4, and so on) still contain additional structure 
that is not visualized here but may contribute non-negligible variance.
The choice of only plotting two PCs is purely optional and for the sake of simplicity, avoiding over-complicated plots that might not be necessary.
Clustering is performed using GMM, which models each cluster as a probabilistic distribution rather than assigning hard boundaries.
</p>
"""


NUTHEAT_CAPTION = """
Cluster-level nutritional profiles visualized as median z-score values. 
The left panel shows absolute nutrient content, while the right panel shows the absolute content values normalized per 100 calories.
See "Insight" for explanation.
"""
NUTHEAT_INSIGHT_TITLE = "Decoding Nutritional Clusters"
NUTHEAT_INSIGHT_SUBTITLE = "How do recipes group by nutrition?"

NUTHEAT_INSIGHT_WHAT = """
<p>
While the PCA landscape shows how recipes are distributed, 
it does not directly explain what defines each cluster.
This is the key limitation of dimensionality reduction, where interpretability is lost when many variables are compressed into two axes.
To bridge this gap, we summarize each cluster using a heatmap of median nutritional values (z-scored).
The heatmap addresses the lost information context from the PCA landscape by summarizing each cluster’s typical nutritional composition to obtain the concrete nutritional profiles.
</p>
"""
NUTHEAT_INSIGHT_WHY = """
<p>
But why comparing absolute values with per-calorie metrics?
</p>
<p>
Because absolute nutritional values alone are strongly influenced by portion size. 
For example, a recipe with very high protein may simply be large in quantity rather than having high protein density.
Introducing per-calorie normalization allows us to separate these two dimensions
by answering two questions: <strong>“how much nutrient does this recipe contain in total?” from the absolute panel, 
and “how much nutritional value do we get per unit of energy?” from the per-calorie panel.</strong> 
This is essential in interpreting the PCA clusters in real-world terms because dietary decisions are rarely based on total intake alone. 
In practice, people care about efficiency, that is when we maximize beneficial nutrients while controlling caloric intake. 
Without this normalization, clusters could be misinterpreted.
</p>
"""
NUTHEAT_INSIGHT_READ = """
<p>
Each row represents a cluster, and each column corresponds to a nutritional feature. 
Each cell represents how a cluster compares to the overall dataset for a given feature.
Values inside the cells are expressed as z-scores of the median within each cluster,
standardized across the clusters, 
allowing comparison with both other clusters and features with different units (see the last section on how to compute the z-score).
This is important to take note because the <strong>z-score does not show absolute unit but rather a comparative values between clusters.</strong>
This means that high positive z-score values (blue cells) indicate that the corresponding features for the said cluster are above average <i>relative</i> to other clusters, 
and vice versa for the negative ones.
Lastly, the heatmap is divided into two panels: the left shows absolute nutritional content, while the right shows normalized values of the absolute content per 100 calories, 
which reflect nutritional density rather than total quantity.
Reading across a row reveals the nutritional “signature” of a cluster.
</p>
"""
NUTHEAT_INSIGHT_FINDINGS = """
<p>
At first glance, it may be hard to see standing-out patterns between the clusters, and that is not surprising.
We already see from the PCA landscape figure that lots of regions filled with overlapping clusters here and there,
however we can still see extreme characteristics from some clusters.
For example, cluster 2 stands out clearly in the absolute content panel where it shows very high values in calories, fat, and fiber in absolute terms (around >1), 
but less nutritious with negative values in sugar, protein, sodium, and even fiber per 100 calories. 
This is important point because only looking at the PCA map tells us that the right-side region is highly nutritious.
That is indeed correct but what it is not telling us is that most of them are large-serving recipes, with obviously more nutritions.
Looking at the heatmap then shows us another interesting perspective where we now know that this cluster maybe represents less dense meals, and indeed, 
we can see in the PCA map that most members are desserts.
Cluster 1 presents a contrasting profile, 
where it has extremely high sugar content (+2.31) while being low in protein and fiber. 
In the PCA map, we can see that most of the recipes in this cluster can be seen fall into the "Dessert" category with dense sugar content,
meaning it may be similar to cluster 2 but with much less serving portions.
Cluster 6 represents another distinct group: high and dense protein and fiber content (from the absolute and per-calorie panels) with moderate calorie levels. 
This suggests that this cluster is highly and effectively nutritious, mostly associated with lean protein or balanced meals.
Once again, from the PCA map, it shows that most of cluster 6's members fall into the meat-like category with high protein.
This is roughly similar to cluster 0 above it with even much more protein content. 
In fact, cluster 0 has the highest protein density, which means <strong>your expensive protein powder can be replaced by almost every recipes within cluster 0 :)</strong>
Lastly, cluster 5 shows low values across most nutrients but a sharp spike in sugar per calorie, 
indicating poor nutritional density which mostly be found in beverages and snacks.
</p>
<p>
All of the above findings are supported and can be easily spotted by the figure below showing the top 5 recipe categories in terms of number of recipes for each cluster.
Immediately we can see some interesting features such as "Desserts" dominating cluster 1, "Beverages" dominating cluster 5, while meat-like products dominating cluster 0.
</p>
"""
NUTHEAT_INSIGHT_INTERPRET = """
<p>
One thing we can take away from this analysis is the distinction between quantity and quality. 
A recipe can appear nutrient-rich in absolute terms simply because it is large, but when normalized per calorie, its nutritional value may be low. 
The heatmap makes this trade-off explicit. For example, some clusters are high in protein but also high in sodium, 
while others achieve high fiber and protein with relatively lower sugar. 
<strong>This highlights that nutrition is inherently multidimensional, and improving one aspect often comes at the cost of another.</strong>
</p>
"""
NUTHEAT_INSIGHT_METHOD = """
<p>
For each cluster identified by the Gaussian Mixture Model, we first compute a representative value 
for every nutritional feature using the <i>median</i> across all recipes within that cluster. 
Let \( x_{c,f} \) denote the median value of feature \( f \) in cluster \( c \).

To enable comparison across features with different units and scales, these values are standardized 
using a <strong>z-score transformation across clusters</strong> (not across individual recipes). 
Specifically, for each feature \( f \), we compute the mean and standard deviation of the cluster-level medians:
</p>

<p style="text-align:center;">
\( \mu_f = {1 \over K} \sum_{c=1}^{K} x_{c,f}, 
\sigma_f = \sqrt{ {1 \over K} \sum_{c=1}^{K} (x_{c,f} - \mu_f)^2 } \)
</p>

<p>
where \( K \) is the number of clusters. The z-score for each cluster-feature pair is then given by:
</p>

<p style="text-align:center;">
\( z_{c,f} = {x_{c,f} - \mu_f \over \sigma_f} \)
</p>

<p>
This transformation centers each feature at zero and rescales it to unit variance across clusters, 
so that positive values indicate clusters with above-average levels of that feature, while negative values 
indicate below-average levels. Because the standardization is performed <em>across clusters</em>, the heatmap 
highlights <strong>relative differences between clusters</strong> rather than absolute magnitudes.

The same procedure is applied separately to both absolute nutritional features and their corresponding 
per-100-calorie version, so that the heatmap consistently reflects both total quantity and 
nutritional density on a comparable scale.
</p>
"""



WINDROSE_CAPTION = """
Each wedge represents a fixed bin of total preparation time, ranging from 0–15 minutes to more than 4 hours, 
arranged clockwise to reflect increasing duration. 
The radial length of each wedge is proportional to the square root of the fraction of recipes in that bin (hover to see the details), 
so that wedge area visually reflects the relative population size while preventing dominance by the largest bins. 
Within each wedge, colors indicate the top recipe categories in that time regime, with segment sizes proportional to their share of recipes inside the bin (see "Insight" for explanation). 
Surrounding the wedges, an outer blue ring encodes the review-count-weighted average rating for each time bin, 
providing a compact summary of user-perceived recipe quality adjusted for sampling reliability. 
The chart simultaneously shows how recipe populations are distributed across cooking time, 
how their category composition shifts with duration, and how these regimes differ in collective user ratings.
Zoom the figure to see the wedges clearer.
"""

WINDROSE_INSIGHT_TITLE = "The Recipe's Compass of Duration"
WINDROSE_INSIGHT_SUBTITLE = "How recipe populations change across total cooking time regimes?"

WINDROSE_INSIGHT_FINDINGS = """
<p>
In this section, the chart is surely much easier to learn than the others. 
It shows the entire recipe dataset (with valid reviews and relevant parameters) distributed across cooking time (or more specifically, the total time, which is the combined cooking plus preparation time).
Each angural sector corresponds to a fixed total-time range, starting from quick recipes for less than 15 minutes to effort-required ones with even more than 4 hours.
The length of each wedge represents how many recipes fall into that time bin (see last section).
Longer wedges indicate time regimes that dominate the dataset while shorter ones correspond to less common regimes.
Within each wedge, we split the color into the top four recipe categories based on the number of recipes.
</p>
<p>
From this, we can immediately see some striking patterns that connect to the culinary behavior.
First, <strong>short-time recipes, particularly those requiring 15-45 minutes, dominate the dataset,</strong>
and the difference is striking with longer ones (recipes requiring more than 1.5 hours do not constitute more than 40% of the combined recipes within 15-45 minutes).
Going more specific, we can start from breakfast that is concentrated almost entirely in the shortest time bin. 
Shifting a bit, lunch/snacks, vegetables, and one-dish meals dominate short- to mid-range duration, indicating practical everyday meal.
What is kind of surprising is that desserts appear to be spanning across multiple wedges being the most dominant category in almost every one of them.
This means that, as it may look simple, turns out dessert can be a complex dish!
Going to longer-time recipes, new categories are starting to appear such as yeast bread, stews, and pork, dishes that require lots of prep time (for marinating for example).
</p>
<p>
Lastly, the thin outer ring encodes the review-weighted average rating for each time bin,
plotted above the wedges to avoid interfering with the main population structure.
The ring shows that <strong>longer cooking time does not necessarily imply higher ratings</strong>.
Instead, ratings tend to remain relatively stable across bins, showing that perceived recipe quality
may be more strongly influenced by final execution and appeal than by time investment alone.
Indeed, exceptional dish is worth the effor.
All this makes the chart a structured view of how culinary types reorganize across time.
</p>
"""

WINDROSE_INSIGHT_METHOD = """
<p>
Each recipe in the dataset is assigned to one of eight fixed total-time bins:
0–15 min, 15–30 min, 30–45 min, 45–60 min, 1–1.5 h, 1.5–2 h, 2–4 h, and &gt;4 h.
These bins define the eight angular wedges of the chart.
If we denote by \(N_i\) the number of recipes in time bin \(i\), and by \(N_{\mathrm{tot}}\) the total number of recipes used in the analysis,
then the radial extent of wedge \(i\) is defined as
\[
r_i = \sqrt{ N_i \over N_{\mathrm{tot}}}
\]
This square-root scaling preserves the ordering of population size across bins, 
while compressing the dynamic range so that very large bins do not visually overwhelm the smaller ones.
In other words, the wedge length represents recipe population share, but in a form that remains readable in a circular plot.
</p>

<p>
The thin outer ring provides the review-count-weighted average rating of recipes in each time bin.
For each recipe \(k\), let \(R_k\) be its mean rating and \(w_k\) be its number of reviews.
Then the weighted average rating for time bin \(i\) is computed as
\[
R_i = {\sum_{k \in i} w_k R_k \over \sum_{k \in i} w_k}
\]
Using review count as the weight gives greater influence to recipes whose ratings are supported by more user feedback,
thereby reducing the visual impact of highly rated recipes that were reviewed only a few times.
This value is encoded not as an additional wedge length, which would compete with the main population axis,
but as a surrounding ring so that rating remains visually secondary but still immediately comparable across bins.
</p>
"""



FEATURE_CAPTION = """
Each circle represents a feature, positioned by how its predictive importance is distributed between the star rating model (Rating), 
the text sentiment model (Sentiment), and the gap model (Gap).
Circle size reflects total SHAP magnitude. 
Features near a corner primarily drive that model's predictions; features in the center contribute broadly across all three. Color indicates feature group.
Look "Insight" for more details. Beware of long texts since this is the most comprehensive analysis in this dashboard.
"""

FEATURE_INSIGHT_TITLE = "The Anatomy of Ratings"
FEATURE_INSIGHT_SUBTITLE = "How would ratings be influenced by recipes and reviewers?"

FEATURE_INSIGHT_CONTEXT = """
<p>
Every recipe in the dataset has two numbers attached to it: a star rating (aggregated over its reviews) and a review count. 
We can assume that these two numbers are the primary signals of recipe quality and popularity. 
If a recipe has a 4.8 average and 500 reviews, the conventional wisdom says it must be excellent, 
whereas if it has a 3.2 average and 12 reviews, it must be mediocre and obscure. 

However, there is a ctach. When a reviewer sits down and writes a review, they are actually performing two separate acts simultaneously.
They write something that describes their experience like what worked and how they feel,  
and then they click a number between one and five stars. 
These two acts are supposed to be measuring the same thing: how good was the recipe. 
But what if they actually don't? 

For example, people are reluctant to give low stars to someone who shared a recipe publicly, because, let's be honest, <strong>it feels mean</strong>, 
so people rates high out of generosity rather than genuine assessment.
In fact, the distribution of star ratings in this dataset is heavily skewed toward five stars; 
look at the figure below where it shows that the vast majority of reviews are five-star ratings with the rest trailing off sharply. 

It is also possible that different people interpret the scale differently. 
One person's (an experienced cook for example) four stars is another person's (a cooking newbie) five stars for an identical experience. 
This is not a normal distribution of quality but rather something that tells you more about social norms than about recipe quality.
Of course, we could not be certain whether the statement of "many 4-5 stars translate to a catalogue of good recipes" is true or not 
because that itself would require a study of causality,
but, lo and behold, that is exactly what are we trying to assess here.

The review text, by contrast, is harder to fake.
 When someone writes "this rendang was edible but nothing special and I probably won't make it again," 
 the words carry the negative signal that the four stars they clicked suppressed. 
 Natural language has the resolution to express nuance, ambivalence, and qualified praise that the integer star scale simply cannot encode. 
 <a href=https://en.wikipedia.org/wiki/Sentiment_analysis>Sentiment analysis</a> can recover this signal and produce a continuous score 
 that reflects what the reviewer actually said rather than what they clicked.
This is the background problem. 
The stars are what everyone uses. The text is what everyone ignores. 
The question is: <strong>how much does this matter, and what specifically is being lost?</strong>
</p>
"""

FEATURE_INSIGHT_EXPECT = """
<p>
The central hypothesis driving this section of analysis is that the star rating and the sentiment score extracted from review text 
are measuring the same underlying experience. 
More specifically, we hypothesized two things. 
<strong>First,</strong> that the star rating would be more strongly predicted by reviewer identity, which is their personal "generosity baseline" (what would their rating knowing their personality) 
and their consistency, than by anything about the recipe itself. 
<strong>Second,</strong> that the sentiment score extracted from text would be more influenced by recipe content than stars are, 
because text is harder to socially influenced than a star click. 
Underlying all two hypotheses is a question that the analysis is really trying to answer: 
<strong>when a recipe gets a high rating, how much of that rating is actually about the recipe?</strong>
</p>

"""

FEATURE_INSIGHT_METHOD = """
<p>
To test these hypotheses, the analysis was conducted at the level of individual reviews rather than recipe averages. 
For each review, three classes of features were assembled simultaneously.
<strong>Recipe features</strong> captured everything about the dish itself: its nutritional content, cooking time, etc. 
<strong>Reviewer features</strong> captured the behavioral history of the person writing the review: 
their personal mean rating across all their other reviews (their generosity baseline), the consistency of their ratings, 
how many reviews they had written in total, and how familiar they were with this specific recipe category. 
<strong>Review context features</strong> captured the situational circumstances of the review: how many days after publication it was written, 
what position it occupied in the sequence of all reviews for that recipe, and how long the text was.
Three separate <a href=https://lightgbm.readthedocs.io/en/latest/>LightGBM machine learning models</a> were then trained on this feature set, 
each predicting a different target. 
Model A, <strong>"Rating"</strong>, predicted the individual star rating. 
Model B, <strong>"Sentiment"</strong>, predicted the sentiment score extracted from the review text using <a href=https://hex.tech/templates/sentiment-analysis/vader-sentiment-analysis/>VADER</a>, 
a sentiment analysis tool that scores text on a continuous scale. 
Model C, <strong>"Gap"</strong>, predicted the gap discussed before (difference between the star rating and the sentiment score) which directly measures 
how much more generously the reviewer scored in stars than their words expressed. 
All three models used the same features and the same methodology so that differences in their outputs are attributable to the targets themselves rather than modeling choices.
After training, <a href=https://en.wikipedia.org/wiki/Shapley_value>SHAP values</a> were computed for each model. 
SHAP itself is a method that takes a trained model and decomposes each prediction into contributions from individual features. 
It answers, for every review, how much did each feature push the predicted value up or down? 
</p>
<p>
One extremely important underlying assumption used here to note is: <strong style="color: #e74c3c; font-weight: 600;"> we assume that the sentiment analysis score from VADER is accurate and that it can be compared (after some proper transformation) directly
to the star rating system.</strong> This assumption will be used for interpretation throughout this whole section, 
as well as for interpretation in the "Reliability" panel (you can click the top-left button to switch panel).
</p>
"""

FEATURE_INSIGHT_FINDINGS_RED = """
<p>
Now, it's time to talk about the ternary plot. First off the reviewer group (red circles).
The dominant feature is "Reviewer Loo Mean", which is the reviewer's mean rating from their reviews. 
Its position tells us that a reviewer's personal generosity baseline 
is primarily a driver of rating gap rather than a driver of star ratings alone. 
This means that a generous reviewer does not write more positive text than others (they just click higher stars regardless of what their text expresses). 
The three smaller reviewer circles means that reviewer consistency (Reviewer Loo Std), experience (Reviewer's Reviews), and category familiarity (Familiarity) 
all play similar roles: 
they contribute to both star predictions and gap predictions, but contribute almost nothing to predicting what a reviewer will write in their text. 
Lastly, the reviewer group as a whole lives on the right side of the ternary, between Rating and Gap, and away from Sentiment. 
<strong>This means reviewer identity is the primary source of the star-sentiment disconnect. It "inflates" stars without inflating text.</strong>
</p>
"""
FEATURE_INSIGHT_FINDINGS_ORANGE = """
<p>
Second is the sentiment group(orange) with only one feature: Sentiment Scaled, the VADER-derived sentiment score from the review text itself. 
The position of the circle is near the Gap corner with essentially zero pull toward Sentiment. 
The zero Sentiment contribution is not surprising because sentiment score cannot be used to predict itself since it is the target variable of the Sentiment model, 
so it was deliberately excluded from the Sentiment model's feature set. 
But the higher Gap contribution compared to Rating is genuinely interesting. 
<strong>The sentiment score is a stronger predictor of inflation than it is of stars.</strong> 
What this means is that when a reviewer's text is genuinely positive, the gap between their star and their text actually widens 
because their star is constrained by the ceiling at five while their sentiment can only go so high, but generous reviewers are already at five stars. 
Conversely, mildly negative text produces a large positive gap because the star stays at four while the sentiment score drops. 
The sentiment score ends up capturing this asymmetric relationship more through the gap than through the raw star.
The circle is also arguably large in size, confirming that Sentiment Scaled is one of the more globally important features in the analysis.
</p>
"""

FEATURE_INSIGHT_FINDINGS_BLUE = """
<p>
The Review Context group contains three features: Review Length, Days Since Publication, and Review Position. 
These are the situational features at which they describe the circumstances under which a review was written.
All three circles sit in the left half of the triangle, pulled toward the Sentiment corner. 
Review Length is the largest of the three and sits closest to the Sentiment corner. 
Long reviews strongly predict sentiment scores (people who write at length tend to be more analytical and their text scores lower on sentiment as a result) 
but review length contributes much less to predicting stars or the gap. 
This makes sense because <strong>the length of what someone writes does not guarantee what they click.</strong>
Days Since Publication sits slightly higher and more toward the center. 
<strong>Reviews written much later after a recipe was published tend to come from cooks who specifically sought the recipe out rather than discovering it casually.</strong 
This changes both what they write and slightly how they rate, but the text effect is stronger.
Review Position sits closest to the center of the three which means that 
<strong>being the hundredth reviewer of a recipe as opposed to the first influences everything slightly.</strong> 
The recipe's reputation has been established, expectations are set, the community has filtered toward people who already expect to like it.
Based on these, what people write is more sensitive to the circumstances of the review than what stars they click. 
<strong>Stars are more habitual and identity-driven. Text is more situational and responsive to context.</strong>
</p>
"""

FEATURE_INSIGHT_FINDINGS_GREEN = """
<p>
The green circles are the recipe group, which contains the largest number of features: 
all the nutritional columns, time features, complexity metrics, etc. 
The most striking pattern from this group is the spread, distributed across almost the entire interior space of the triangle. 
Different recipe attributes play fundamentally different roles in the three models, 
and <strong>there is no single story for "recipe features" the way there is for "reviewer features" or "context features."</strong>
What is more interesting, there is a visible diagonal band of features running from the Sentiment corner toward the middle between Gap and Rating.
This means most recipe features split their importance roughly between Rating and Gap with a moderate Sentiment contribution. 
They matter for predicting stars and they contribute to gap, but they also weakly predict text sentiment.
</p>
"""

# FEATURE_INSIGHT_SUMMARY = """
# This plot reveals that the three types of information a review carries:  
# who wrote it, what the recipe is, and the circumstances of the writing,  
# play fundamentally different roles in determining stars vs. sentiment vs. gap between them. 
# Reviewer identity, dominated overwhelmingly by a single feature representing the reviewer's personal generosity baseline, sits near the Gap corner, 
# meaning that <strong>knowing who is rating a recipe tells you more about how different their star will be relative to their own words 
# than it tells you about what they will actually write.</strong> 
# Recipe content spreads broadly across the entire triangle with no dominant corner, meaning <strong>different recipe attributes drive different outcomes, 
# but no recipe attribute comes close to matching the strong influence of reviewer identity.</strong> 
# Review context features lean toward the Sentiment corner, 
# meaning that <strong>situational factors like how long a review is and when it was written shape text more than they shape stars.</strong> 
# Most interestingly, the Sentiment corner itself is the emptiest part of the triangle across all feature groups, 
# which means that nothing in the dataset is primarily a driver of expressed text sentiment. 
# This is one of the very important finding: <strong>the review text is the most honest and least structured of the three signals:</strong> 
# stars are largely predictable from who is clicking them, 
# gap is predictable from reviewer generosity bias, 
# but what people actually write in their reviews resists any single dominant explanation 
# and is therefore the closest thing this dataset has to an unmanipulated record of genuine recipe experience.
# """

FEATURE_INSIGHT_SUMMARY = """
The plot below summarizes the ternary plot findings nicely.
It takes the SHAP values from each model and groups them by feature class: Recipe, Reviewer, Sentiment, and Review Context, 
then expresses each group's contribution as a percentage of the total SHAP magnitude. 
This is the figure that directly answers the central question of the analysis.
For the Star Rating model, the Reviewer class accounts for 42.7% of total SHAP magnitude, 
Recipe features account for 26.9%, Sentiment accounts for 21.1%, and Review Context accounts for 9.3%. 
Summarizing all:
the person doing the rating contributes more to the star than the recipe being rated does. 
This is not a subtle effect because there is a 16 percentage point gap between the reviewer's contribution and the recipe's contribution to the same star rating. 
<span style="
  color: #e74c3c;
  font-weight: 600;
  border-bottom: 2px solid #e74c3c;
  padding-bottom: 2px;
">
When you give a recipe some stars, roughly 43% of what determined that number was something about you rather than something about the recipe 
</span>
(I should highlight this because this might be the single most important finding from the entire analysis of this dashboard).
</p>
<p>
The Sentiment model reverses this to a meaningful degree. Review Context (dominated by review length and timing) accounts for 41% of sentiment variance, 
Reviewer features account for 34.1%,  
Recipe features account for 25%. 
The recipe's contribution to sentiment is slightly lower than its contribution to stars in absolute percentage terms, 
but the reversal of the reviewer versus context dominance is notable. 
Sentiment scores are being driven more by situational and behavioral context such as how long did you write and when did you review, rather than by stable reviewer identity. 
This is consistent with the scenario where a generous reviewer will still click five stars consistently, but what they write in the box varies more with the actual experience.
</p>
<p>
From the Gap model, Sentiment accounts for 34.3% of gap variance while Reviewer accounts for 34.0%; they are nearly tied. 
Recipe accounts for 22.9%. This means rating gap is split almost equally between what the text said and who the reviewer is. 
Reviewers who are generous clickers produce the most gap, 
but sentiment that is genuinely positive also pulls the gap because positive text is paired with a five-star ceiling. 
The recipe contributes about a fifth of the gap, meaning some recipe types systematically attract inflated ratings beyond what their sentiment would predict.
</p>
"""

FEATURE_INSIGHT_INTERPRET = """
<p>
The results above may be very interesting, yes, but there is one extremely important point.
For each model, we compute the R² value, and they are shown in the KPI cards at the bottom.
This value measures how much of the variance in a target variable a model can explain using the features it was given. 
A value of 1 means the model predicts perfectly whilst a value of 0 means the features carry no useful information about the target whatsoever. 
In this analysis, R² tells us something specific and important: it quantifies how much of the reason a reviewer gives a particular star, 
writes a particular sentiment, or produces a particular gap is determined by things we can measure: the features. 
The remaining fraction, what R² does not explain, represents variance driven by things outside the feature set; something like mood, cooking skill on the day, etc. 
With that framing, the three R² values become a map of how predictable each signal is.
</p>
<p>
First, we see that the Star model explains 52% of the individual rating variance. 
This means that more than half of why a reviewer gives the stars they give is predictable from who they are, 
what recipe they are reviewing, and when they reviewed it. 
The ternary plot tells us exactly why: reviewer identity, 
particularly the "Reviewer Loo Mean", is so dominant and so stable that it makes star-clicking behavior largely foreseeable. 
The Gap model's R² of 0.56 is even higher, which at first seems surprising but makes complete sense in light of the ternary findings. 
The gap is the most structured and learnable signal of the three precisely because it is driven by the same reviewer personality trait that dominates the Star model, 
now measured more cleanly without the recipe signal mixed in.

The Sentiment model's R² of 0.23 is the number that matters most for interpretation, 
and it tells the most important story of the three. After assembling the richest possible feature set, including 
everything about the recipe, everything about the reviewer's history, and everything about the review context,  
the model can only explain 23% of why a reviewer's text scores the way it does. The other 77% is not captured by any of these features. 
This means that what people write in their reviews is genuinely responsive to the actual experience 
in ways that the structured features in this dataset cannot fully anticipate or encode. 
Stars are predictable because they are, at least based on these findings, socially-driven and reviewer-dependent. 
Sentiment resists prediction because it is the signal that is still, 
imperfectly but meaningfully, trying to describe what actually happened in the kitchen. 
<strong>The 29-percentage-point gap between the Star R² and the Sentiment R² is therefore the quantitative size of the space where honest recipe experience lives; 
the space that the star rating system, by being so predictable, has largely abandoned.</strong>
</p>
"""

RELIABLE_CAPTION = """
Pearson 𝑟 measures the correlation between individual star ratings and VADER sentiment scores within each recipe category. 
Higher values indicate that stars and expressed text sentiment are more aligned so that the rating system is more honest. 
Lower values indicate decoupling between what reviewers write and what they click. 
No category reaches 𝑟 = 0.5, suggesting moderate-to-weak alignment universally across the dataset.
"""

RELIABLE_INSIGHT_TITLE = "Category Reliability"
RELIABLE_INSIGHT_SUBTITLE = "Where star ratings match the written review most closely?"

RELIABLE_INSIGHT_WHAT = """
The Pearson 𝑟 values shown here measure how honestly the star rating tracks expressed text sentiment within each recipe category. 
A higher 𝑟 means stars and text are more aligned, 
meaning that when someone writes positively in that category they also tend to give high stars, 
and when they write critically the stars fall accordingly. 
A lower 𝑟 means the two signals are conflicting: stars are being driven by something other than what the text actually expresses.
"""

RELIABLE_INSIGHT_FINDINGS = """
What is plotted here is the top 30 categories with highest 𝑟 value.
The first thing to notice across all three images is that every single category, without exception, falls between roughly 0.2 to 0.4. 
Not a single category reaches 𝑟=0.5, which would be the threshold for reliable alignment between stars and text (this threshold is somewhat arbitrary and I choose 0.5). 
This means that <strong>the star rating system is only moderately honest in the best case, and could even be less honest.</strong>

Looking at the top, the categories where stars and sentiment are most aligned are Yeast Breads, One Dish Meal, Chicken, Chicken Breast, Pork, Meat, Steak, and Pie. 
These are predominantly protein-rich, savory, or technically demanding categories where cooks tend to have expectations before they start cooking. 
A chicken breast recipe either works or it does not, and the feedback loop between outcome and expressed opinion is relatively tight. 
When these cooks write positively they mean it, and when they click five stars it is more likely to reflect genuine satisfaction than social generosity. 

The bottom eight categories are Beverages, Under 30 Minutes, Under 15 Minutes, Low Protein, Spreads, Beans, Potato, and Sauces. 
These categories can be inherently ambiguous and variable in what a good outcome even looks like. 
For example, a beverage recipe can mean anything from a smoothie to a cocktail to a hot drink, and cooks who try these recipes bring wildly different expectations and contexts. 
The time-constraint categories are particularly interesting, in a way that 
these are defined entirely by convenience rather than cuisine (or so we think), 
meaning reviewers chose the recipe primarily because it was fast, not because they had an expectation of what the dish should taste like. 
<strong>When expectations are poorly formed, the gap between what someone writes and what they click widens, 
because the star becomes more of a social gesture and the text becomes more of a genuine reflection on whether the promised convenience was delivered.</strong>

"""

RELIABLE_INSIGHT_INTERPRET = """
The full picture tells us something meaningful.
Categories with well-defined quality standards like bread should rise, chicken should be moist, 
steak should be cooked to the right temperature, etc. produce more honest rating behavior than categories defined by convenience, ingredient type, 
or dietary constraint where the quality benchmark is harder to define. 
<strong>The practical implication is that any recipe rating in a low-reliability category should be treated 
with considerably more skepticism than the same star rating in a high-reliability category, 
because the probability that the star reflects genuine assessed quality rather than social generosity is meaningfully lower.</strong>
"""

