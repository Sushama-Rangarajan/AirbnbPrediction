
```{r message=FALSE, warning=FALSE, include=FALSE}
pacman::p_load(tidytext,qdap, lubridate,dplyr,textdata, textcat,cld2,sentimentr,tm, tokenizers,  quanteda , ggthemes,SnowballC,cld3,lexicon, fpp3, GGally, gridExtra,rJava,mallet, topicmodels, tidyr,stringr, ggplot2, wordcloud,rjson,rgdal,sqldf, caret,e1071,corrplot, xgboost,gbm, vtreat,lubridate, reshape2,data.table, neuralnet, GGally,pROC,lime, furrr, parallel, mltools, randomForest, ranger, lm.beta)
```

```{r message=FALSE, warning=FALSE, include=FALSE}
air_listings <- read.csv('Jan_listings 2.csv', stringsAsFactors = FALSE)
air_listings <-  subset(air_listings, !air_listings$calendar_updated %like% "never")
columns <- c("id","host_since","house_rules", "host_is_superhost","host_response_rate","host_response_time",
             "host_has_profile_pic","host_identity_verified", "neighbourhood_cleansed","neighbourhood_group_cleansed",
             "is_location_exact","property_type", "room_type","accommodates","bedrooms","bathrooms","beds",
             "price", "security_deposit","cleaning_fee","minimum_nights","maximum_nights",
             "availability_30","availability_60","availability_90","availability_365",
             "number_of_reviews","review_scores_location","review_scores_value", 
             "instant_bookable","cancellation_policy",
             "calculated_host_listings_count","reviews_per_month","guests_included","require_guest_profile_picture",
             "require_guest_phone_verification")
bnb <- air_listings[,names(air_listings) %in% columns]
```

```{r message=FALSE, warning=FALSE, include=FALSE}
#Checking host years with airbnb
bnb$host_since <- as.POSIXct(bnb$host_since, format = "%Y-%m-%d")
bnb$host_since <- lubridate::year(bnb$host_since)
bnb$host_experience <- 2020 - bnb$host_since
bnb <- bnb[,-3]

#converting prices to numeric
bnb$price <- gsub("[$]",'',bnb$price)
bnb$price <- as.numeric(gsub(',','',bnb$price))
bnb <- subset(bnb, bnb$price != 0)
bnb <- subset(bnb, !is.na(bnb$host_experience))
bnb$security_deposit <- gsub("[$]",'',bnb$security_deposit) 
bnb$security_deposit  <- as.numeric(gsub(',','',bnb$security_deposit))

bnb$cleaning_fee <- gsub("[$]",'',bnb$cleaning_fee)
bnb$cleaning_fee <- as.numeric(gsub(',','',bnb$cleaning_fee))

#converting factors
bnb[,c(5:12,30:33)] <- lapply(bnb[,c(5:12,30:33)],factor)
bnb <- bnb[,-c(4,6)]

```

```{r   message=FALSE, warning=FALSE, include=FALSE}
#Treating NAs
#Listings that doesn't have bathrooms or bedrooms refer those types of listings with studio apartment. Hence will be replaced as 1

bnb$bedrooms[is.na(bnb$bedrooms)] <- 1
bnb$bathrooms[is.na(bnb$bathrooms)] <- 1
bnb$bedrooms[bnb$bedrooms == 0] <- 1
bnb$bathrooms[bnb$bathrooms == 0] <- 1

bnb$beds <- ifelse(is.na(bnb$beds),bnb$bedrooms,bnb$beds)
bnb$beds[bnb$beds == 0] <- 1

#No security deposit and no cleaning fee is replaced with 0
bnb$security_deposit[is.na(bnb$security_deposit)] <- 0
bnb$cleaning_fee[is.na(bnb$cleaning_fee)] <- 0


#listings that has no reviews will not have review scores and is thus replaced with 0 
bnb$review_scores_location[is.na(bnb$review_scores_location)] <- 0
bnb$review_scores_value[is.na(bnb$review_scores_value)]<- 0
bnb$reviews_per_month[is.na(bnb$reviews_per_month)] <- 0

```

```{r   message=FALSE, warning=FALSE, include=FALSE}
#Creating Property buckets and factor for cancellation policy and response time
bnb$property_bucket <- case_when(
   bnb$property_type %in% "Apartment" ~ "Apartment",
   bnb$property_type %in% "House" ~ "House",
   bnb$property_type %in% "Condominium" ~ "Condominium",
   bnb$property_type %in% "Loft" ~ "Loft",
   bnb$property_type %in% "Serviced apartment" ~ "Serviced apartment" ,
   bnb$property_type %in% "Boutique" ~ "Boutique",
   TRUE ~ "Other"
 )

bnb$host_response_time[bnb$host_response_time == "N/A"] <- "Not Applicable"
bnb[,c(3,35)] <- lapply(bnb[,c(3,35)], factor)

```

```{r   message=FALSE, warning=FALSE, include=FALSE}
nc <- bnb[,c("neighbourhood_cleansed","neighbourhood_group_cleansed","id")]
nc <- nc %>% count(neighbourhood_cleansed, sort = TRUE)
nc1 <- top_n(nc,20)
nc1$neighbourhood_cleansed <- as.character(nc1$neighbourhood_cleansed)
nc1 <- nc1[,-2]

bnb$Neigh_cluster <- as.character(bnb$neighbourhood_cleansed)
bnb$Neigh_cluster[!bnb$Neigh_cluster %in% nc1] <- "Other"
bnb$Neigh_cluster <- as.factor(bnb$Neigh_cluster)

```

```{r   message=FALSE, warning=FALSE, include=FALSE}

#Decoding the House rules
house_rules <- bnb[,c("id","house_rules")]
stop_word <- stop_words %>% filter(lexicon == "snowball")
house_rules$house_rules <- str_replace_all(house_rules$house_rules, "[^[:alnum:]]", " ")

 house_rules1 <- house_rules %>% unnest_tokens(bigrams, house_rules, token = "ngrams", n = 2)
house_rules1 <- house_rules1 %>%separate(bigrams, c("word1","word2"), sep = " ")
hr <- house_rules1 %>% group_by(id) %>% filter(word1 %in% c("no","non","not"))
hr$word2 <- str_replace_all(hr$word2, c("smoke","smoking","smoker","smocking"),"smoking")
hr$word2 <- str_replace_all(hr$word2, c("pets","pet","petss","animals","animal","cats","dogs","cat","dog"),"pets")
hr$word2 <- str_replace_all(hr$word2, c("drinking","drink","alcohol","alcoholic","alcoholics"),"drinking")
hr$word2 <- str_replace_all(hr$word2, c("party","partying","parties","partie","partiers","partys"),"partying")
hr$word1 <- str_replace_all(hr$word1, "non","no")
hr$combined <- paste(hr$word1,hr$word2)
hr <- subset(hr, hr$combined %in% c("no smoking","no drinking","no pets", "no partying"))
hr <- hr[,c(1,4)]
house_r <- dcast(hr,id~combined,length,fill = 0)

#Merging this with parent dataset
bnb <- bnb %>% left_join(house_r, by = c("id" = "id"))
bnb[is.na(bnb)] <- 0
bnb$`no smoking`[bnb$`no smoking` > 1] <-  1
bnb$`no pets`[bnb$`no pets` > 1] <- 1
bnb$`no partying`[bnb$`no partying` > 1] <-1
bnb <- bnb[,-2]
bnb[,c(36:39)] <- lapply(bnb[,c(36:39)], factor)
```

```{r   message=TRUE, warning=FALSE, include=FALSE}
#Creating Stay type buckets : Long term, short terms
bnb <-  bnb %>% mutate(Stays = ifelse(minimum_nights <=7,"Few_Days",
                                      ifelse(minimum_nights <=28,"Few_Weeks",
                                             ifelse(minimum_nights<=180,"Few_Months","Long_Term"))))
                         
bnb$Stays <- as.factor(bnb$Stays)
```


```{r   message=FALSE, warning=FALSE, include=FALSE}
amenities <- air_listings %>% inner_join(bnb, by = c("id" = "id"))
amenities <- amenities[,c(1,59)]

amenities$amenities <- str_replace_all(amenities$amenities,"[{}]","")
amenities$amenities <- str_replace_all(amenities$amenities,'"',"")

amenities <- data.table(amenities)
am1 <- amenities[, strsplit(amenities, ",", fixed = TRUE),by=id]

amn1<- am1 %>% count(V1,id) 
#Most commonly offered amenities

tot_amn <- am1 %>% count(V1, sort = TRUE)
tot_amn <- tot_amn %>% filter(!V1 %like% "^translation missing")
tot_amn <- data.table(tot_amn)
wordcloud(words = tot_amn$V1, freq = tot_amn$n, min.freq = 500, max.words = 50, random.order = FALSE,
          scale=c(2,0.7),colors=brewer.pal(8, "Dark2"),rot.per = 0.2)

top_20 <- top_n(tot_amn,20)
top_20 <- top_20$V1
#Casting each amenity into different columns 
amenities <- dcast(am1[V1 %in% top_20],id~V1,length,fill = 0)

```

```{r   message=FALSE, warning=FALSE, include=FALSE}
#Merging with the parent dataset bnb 
bnb_amn <- bnb %>% left_join(amenities, by = c("id" = "id"))
bnb_amn[is.na(bnb_amn)] <- 0
bnb_amn$TV[bnb_amn$TV > 1] <- 1
names(bnb_amn) <- make.names(names(bnb_amn), allow_ = FALSE)
```

```{r   message=FALSE, warning=FALSE, include=FALSE}

#Auto cancellation

rev <- read.csv("Jan_reviews 2.csv", stringsAsFactors = FALSE)
cancelled <- subset(rev, rev$comments %like% "This is an automated posting.")
auto_can <- cancelled %>% group_by(listing_id) %>% summarise(auto_cancellation = n())
bnb_amn <- bnb_amn %>% left_join(auto_can, by = c("id" = "listing_id"))
bnb_amn$auto_cancellation[is.na(bnb_amn$auto_cancellation)] <- 0
```

```{r   include=FALSE}
#Transit instructions
tr <- air_listings[,c(1,10,12)]
tr$neighborhood_overview[tr$neighborhood_overview == ""] <- NA
tr$transit[tr$transit == ""] <- NA
tr$neighbourhood_desc <- ifelse(is.na(tr$neighborhood_overview), 0,1)
tr$transit_inst <- ifelse(is.na(tr$transit),0,1)
tr <- tr[,c(1,4:5)]
bnb_amn <- bnb_amn %>% left_join(tr, by = c("id"="id"))
names(bnb_amn) <- make.names(names(bnb_amn), allow_ = FALSE)
```


```{r   message=FALSE, warning=FALSE, include=FALSE}
corrcheck <- bnb_amn[,-c(1:9,27:30,34:60,62,63)]
correl <- cor(corrcheck,use = "pairwise.complete.obs")
corrplot(correl)
#Removing correlated variables and other redundant variables from dataset
#Since bedrooms, beds and accommodates will naturally be correlated, those are still retained in the model. 
#Review scores - location is removed but value is retained for modeling
#To check the effect of availability on price, availability 90 and 365 are retained and the others dropped. 
bnb_amn <- bnb_amn[,-c(5,8,29,20,21,25)]
bnb_amn[,c(35:54, 56:57)] <- lapply(bnb_amn[,c(35:54, 56:57)],factor)

```
# Exploratory Analysis


```{r   message=FALSE, warning=FALSE, include=FALSE}
calendar <- read.csv("Jan_calendar.csv")
calendar$date <- as.POSIXct(calendar$date, format = "%Y-%m-%d")
calendar$available = as.factor(calendar$available)
calendar <- calendar %>% inner_join(bnb_amn, by = c("listing_id" = "id"))
cal1 <- calendar[,c(2,5,11, 29)]
cal1$Day <- lubridate::wday(cal1$date, label=TRUE) 
cal1$month <- lubridate::month(cal1$date)
cal1$adjusted_price <- gsub("[$]",'',cal1$adjusted_price)
cal1$adjusted_price <- as.numeric(gsub(',','',cal1$adjusted_price))
cal1 <- cal1 %>% filter(!is.na(adjusted_price))

```

```{r echo=FALSE, message=FALSE, warning=FALSE, out.height="40%", out.width=}
neigh_weekday <- cal1 %>% group_by(neighbourhood.group.cleansed, month,Day) %>% summarise(avgprice = mean(adjusted_price))
ggplot(mapping = aes(x=month, y= avgprice, colour = Day),data = neigh_weekday) + geom_line() +
  facet_wrap(~ neighbourhood.group.cleansed, scales = "free_y") + 
  labs(x = "Month", y = "Average Price", colour = "Day")
``


```{r echo=FALSE, message=FALSE, warning=FALSE, out.width= "50%", out.height="50%"}
wordcloud(words = tot_amn$V1, freq = tot_amn$n, min.freq = 500, max.words = 50, random.order = FALSE,
          scale=c(2,0.7),colors=brewer.pal(8, "Dark2"),rot.per = 0.2)
```

```{r echo=FALSE, message=FALSE, warning=FALSE, out.width= "80%", out.height="60%"}
cancel_price <- cal1 %>% group_by(cancellation.policy,month) %>% summarise(avgprice = mean(adjusted_price))
ggplot(mapping = aes(x=month, y= avgprice,colour = cancellation.policy),data = cancel_price) + geom_line() + 
  labs(x = "Month", y = "Average Price", colour = "Cancellation")
```



# Analysis

## A. Predicting prices using different features for the entire listings 

In this model, multivariate linear regression was used to predict log(price) taking factors such as beds, baths, property types, amenities, cleaning fee etc. into consideration. 

The results of the regression is attached in Appendix I. 

```{r message=FALSE, warning=FALSE, include=FALSE}
regmodel <- bnb_amn
regmodel$logprice <- log(regmodel$price)
regmodel$cleaningfee <- log(regmodel$cleaning.fee + 1)
regmodel$securityfee <- log(regmodel$security.deposit + 1)

regmodel <- regmodel[,-c(1,12:14)] 
```

```{r message=FALSE, warning=FALSE, include=FALSE}
set.seed(121)
regi <- createDataPartition(regmodel$logprice, p = 0.8, list = FALSE)
train_t <- regmodel[regi,]
test_t <- regmodel[-regi,] 
test_t1 <- test_t[,-54]
lm_trn <- lm(logprice~., data = train_t)

#checking std beta to look at feature importance. 
std.coff <- lm.beta(lm_trn)
std.feat <- std.coff$model

lm.predtrn <- predict(lm_trn,test_t1)
lmt.ac_pr <- data.frame(cbind(actual=test_t$logprice, predicted = lm.predtrn )) 
lmt.ac_pr <- lmt.ac_pr %>% mutate(error = predicted - actual)
lmt.ac_pr <- lmt.ac_pr %>% mutate(sqerror = error*error)
```

```{r echo=FALSE, message=FALSE, warning=FALSE, paged.print=TRUE}
paste0("The Model R-Squared is = ", signif(summary(lm_trn)$r.squared,4))
paste0("The Prediction(test) Mean Squared Error is = " , signif(mean(lmt.ac_pr$sqerror),4))
```


## B. Predicting the prices of Queens Neighborhood

This analysis was carried out to understand if the negative reviews had an impact on pricing.  
Sentiment analysis using polarity scores was used to classify reviews as positive and negative, which was then included in the model for prediction. 
The Queens neighborhood locations are grouped into 3 Categories - Astoria, Long Island City and Others

The review polarity graph is shown below : 

```{r   message=FALSE, warning=FALSE, include=FALSE}
#This segment of code takes longer time to run. 
bnb_ids <- bnb_amn %>% filter(neighbourhood.group.cleansed == "Queens")
bnb_ids <- bnb_ids[,1]
rev$date <- as.POSIXct(rev$date, format = '%Y-%m-%d')
rev1 <- rev %>% filter(listing_id %in% bnb_ids)
rev1$year <- lubridate::year(rev1$date)
rev1 <- rev1 %>% filter(year == 2019)
rev1 <- rev1[,c(1,6)]
rev1$comments <- gsub("[^\x20-\x7E]", "", rev1$comments)
rev1 <- rev1 %>% mutate(cld2 = cld2::detect_language(text = comments, plain_text = FALSE),cld3 = cld3::detect_language(text = comments))

#filtering English comments categorized as "en" in both the methods 
rev1 <- rev1 %>% filter(cld2 == "en" & cld3 == "en")
rev1 <- rev1 %>% mutate(id = row_number())
rev2 <- rev1[,c(2,5)]
rev2 <- rev2 %>% unnest_sentences(sentences,comments)
rev2$sentences <- removePunctuation(rev2$sentences)
queen_polarity <- qdap::polarity(rev2$sentences)
```

```{r echo=FALSE, message=FALSE, warning=FALSE, out.width= "60%"}
ggplot(queen_polarity$all, aes(x = polarity, y = ..density..)) + 
  geom_histogram(binwidth = 0.25, fill = "#bada55", colour = "grey60") +
  geom_density(size = 0.75) +
  theme_gdocs() 
```

```{r   message=FALSE, warning=FALSE, include=FALSE}
rev2$polarity <- queen_polarity$all$polarity
pos_terms <- rev2 %>%
  mutate(polarity = queen_polarity$all$polarity) %>%
  filter(polarity > 0) %>%
  pull(sentences) %>% 
  paste(collapse = " ")

neg_terms <- rev2 %>%
  mutate(polarity = queen_polarity$all$polarity) %>%
  filter(polarity < 0) %>%
  pull(sentences) %>% 
  paste(collapse = " ")

all_corpus <- c(pos_terms, neg_terms) %>% 
    VectorSource() %>% 
  VCorpus()

all_tdm <- TermDocumentMatrix(
  all_corpus, 
  control = list(
    weighting = weightTfIdf, 
    removePunctuation = TRUE
  )
)

all_tdm_m <- as.matrix(all_tdm)
colnames(all_tdm_m) <- c("positive","negative")


```

Comparison cloud of positive and negative reviews are shown below : 

```{r echo=FALSE, message=FALSE, warning=FALSE, out.width= "50%", out.height="40%"}
comparison.cloud(
  all_tdm_m,
  max.words = 40,
   scale=c(2,.5),
  colors = c("darkgreen","darkred")
)

```

```{r   message=FALSE, warning=FALSE, include=FALSE}
queens <- rev2 %>% group_by(id) %>% summarise(polarityscore = sum(polarity))
rev1 <- rev1 %>% left_join(queens, by = c("id" = "id"))
rev1$polarityscore[is.nan(rev1$polarityscore)] <- 1
queen_pol <- rev1[,c(1,5:6)]
queen_pol$review <- ifelse(queen_pol$polarityscore >=0, "positive","negative")
queen_pol <- dcast(queen_pol, listing_id ~ review, length, fill =0)
bnb_queens <- bnb_amn %>% filter(neighbourhood.group.cleansed == "Queens")
bnb_queens <- bnb_queens %>% left_join(queen_pol, by = c("id" = "listing_id"))
bnb_queens[is.na(bnb_queens)] <- 0
```

## 1. Linear Regression 

```{r   include=FALSE}
#Queens regression

queen.reg <- bnb_queens
queen.reg$logprice <- log(queen.reg$price)
queen.reg$cleaning <- log(queen.reg$cleaning.fee + 1)
queen.reg$security <- log(queen.reg$security.deposit +1)
queen.reg <- queen.reg[,-c(1,5,12:14)]

set.seed(121)
qi <- createDataPartition(queen.reg$logprice, p = 0.8, list = FALSE)
qtrain <- queen.reg[qi,]
qtest <- queen.reg[-qi,]
qtestx <- qtest[,-59]

qlin <- lm(logprice ~. , data = qtrain)

pred_test <- predict(qlin, qtestx)
lmerr <- data.frame(cbind(actual = qtest$logprice, predicted = pred_test))
lmerr <- lmerr %>% mutate(error = predicted - actual)
lmerr <- lmerr %>% mutate(sqerror = error * error)

```

```{r echo=FALSE, message=FALSE, warning=FALSE}
paste0("The R-squared using Linear Regression is = " , signif(summary(qlin)$r.squared,4))
paste0("The Prediction(test) MSE using Linear Regression is = " , signif(mean(lmerr$sqerror),4))
```

## 2. Gradient Boosting Machines (GBM)

The GBM specification used after tuning the model is as follows :

```{r   include=FALSE}
#GBM - takes time to run
gbm.fit <- gbm(
  formula = logprice ~ .,
  distribution = "gaussian",
  data = qtrain,
  n.trees = 4734,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 10,
  n.cores = NULL, 
  verbose = FALSE
  )  

print(gbm.fit)
gbm.perf(gbm.fit, method = "cv")
```

```{r echo=FALSE, message=FALSE, warning=FALSE}
paste0("Optimal number of trees = ", gbm.fit$n.trees)
paste0("Interaction depth = ", gbm.fit$interaction.depth)
paste0("Shrinkage = ", gbm.fit$shrinkage)
paste0("Cross Validation Folds = ", gbm.fit$cv.folds)
```

```{r echo=FALSE ,message=FALSE, warning=FALSE, fig.width= 8}

set.seed(121)
nran <- sample(1:nrow(qtest),2, replace = FALSE)
check_gbm <- qtestx[nran, ]


model_type.gbm <- function(x, ...) {
  return("regression")
}

predict_model.gbm <- function(x, newdata, ...) {
  pred <- predict(x, newdata, n.trees = x$n.trees)
  return(as.data.frame(pred))
}

# apply LIME
explainer <- lime(qtrain, gbm.fit)
explanation <- lime::explain(check_gbm, explainer, n_features = 10)
plot_features(explanation)

```

```{r echo=FALSE, message=FALSE, warning=FALSE}
#GBM prediction
testgbm <- predict(gbm.fit, qtestx)
gbmerr <- data.frame(cbind(actual = qtest$logprice, predicted = testgbm))
gbmerr <- gbmerr %>% mutate(error = predicted - actual)
gbmerr <- gbmerr %>% mutate(sqerror = error * error)
paste0("The prediction MSE using GBM is ", signif(mean(gbmerr$sqerror),4))

```

## 3. Random Forest

The random forest specification after tuning the model is shown below : 

```{r   message=FALSE, warning=FALSE, include=FALSE}
set.seed(121)
rf <- randomForest(logprice ~., data = qtrain)

rfpred <- predict(rf, qtestx)
rfpred1 <- data.frame(cbind(actual = qtest$logprice, predicted = rfpred))
rfpred1 <- rfpred1 %>% mutate(error = predicted - actual)
rfpred1 <- rfpred1 %>% mutate(sqerror = error * error)

#tuning
hyper_grid <- expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = logprice ~ ., 
    data            = qtrain, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

```

```{r echo=FALSE, message=FALSE, warning=FALSE}
#Tuned model parameters
  optimal_ranger <- ranger(
    formula         = logprice ~ ., 
    data            = qtrain, 
    num.trees       = 500,
    mtry            = 20,
    min.node.size   = 3,
    sample.fraction = .8,
    importance      = 'impurity'
  )

imp <- data.frame(optimal_ranger$variable.importance) 

imp <- imp %>% arrange(desc(optimal_ranger.variable.importance))

setDT(imp, keep.rownames = TRUE)

paste0("Number of trees = ", optimal_ranger$num.trees)
paste0("Number of variables randomly sampled as candidates at each split (mtry) = ", optimal_ranger$mtry)
paste0("Depth of the tree = ", optimal_ranger$min.node.size)

```


The graph indicates the variable importance as observed via Random Forest algorithm.

```{r echo=FALSE, message=FALSE, warning=FALSE, out.width= "60%", fig.align = 'center'}
imp %>% 
top_n(15) %>%
ggplot(aes(reorder(rn,optimal_ranger.variable.importance),optimal_ranger.variable.importance)) +
geom_col() +
coord_flip() +
ggtitle("Top 15 important variables using Random Forest") + 
  labs(y = "Variable Importance" , x= "Features")

```

```{r echo=FALSE, message=FALSE, warning=FALSE}
rftuned1 <- predict(optimal_ranger, qtestx)
rftuned <- data.frame(cbind(actual = qtest$logprice, predicted = rftuned1$predictions))
rftuned <- rftuned %>% mutate(error = predicted - actual)
rftuned <- rftuned %>% mutate(sqerror = error * error)
paste0("The prediction MSE using Random Forest is ", signif(mean(rftuned$sqerror),4))
```

#### Comparison table of all three models are shown below :  

```{r echo=FALSE, message=FALSE, warning=FALSE, paged.print=TRUE}
ovr <- matrix( c(signif(mean(lmerr$sqerror),4) , signif(mean(gbmerr$sqerror),4)  ,signif(mean(rftuned$sqerror),4)), ncol = 3, byrow = TRUE)
rownames(ovr) <- "MSE values"
colnames(ovr) <- c("Linear Model", "Gradient Boost", "Random Forest")
ovr
```


# Sources

* Datasets downloaded from http://insideairbnb.com/get-the-data.html
* Sentiment Analysis in R from https://learn.datacamp.com/courses/sentiment-analysis-in-r 


