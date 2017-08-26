# Analysis of Panic and Alert on Social Media
### Final Project // Advanced Methods in Natural Language Processing @TAU // Spring 2017

Some examples for Named Entity Recognition:
```
Forest fire near La Ronge Sask. Canada
Forest fire near La Ronge Sask. Canada
Named Entities: ['Forest', 'La Ronge Sask']

All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected
All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected
Named Entities: ['shelter', 'shelter']

13,000 people receive #wildfires evacuation orders in California
13,000 people receive wildfires evacuation orders in California
Named Entities: ['California']

Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school
Just got sent this photo from Ruby Alaska as smoke from wildfires pours into a school
Named Entities: ['Ruby Alaska']

#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires
Rocky Fire Update => California Hwy. 20 closed in both directions due to Lake County fire - C Afire wildfires
Named Entities: ['Rocky Fire Update', 'California', 'Lake County']

Apocalypse lighting. #Spokane #wildfires
Apocalypse lighting. Spokane wildfires
Named Entities: ['Spokane']

#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas
flood disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas
Named Entities: ['Manitou', 'Colorado Springs']

Typhoon Soudelor kills 28 in China and Taiwan
Typhoon Soudelor kills 28 in China and Taiwan
Named Entities: ['China', 'Taiwan']

Haha South Tampa is getting flooded hah- WAIT A SECOND I LIVE IN SOUTH TAMPA WHAT AM I GONNA DO WHAT AM I GONNA DO FVCK #flooding
Haha South Tampa is getting flooded hah- WAIT A SECOND I LIVE IN SOUTH TAMPA WHAT AM I GONNA DO WHAT AM I GONNA DO FVCK flooding
Named Entities: ['South Tampa', 'SOUTH TAMPA']

#raining #flooding #Florida #TampaBay #Tampa 18 or 19 days. I've lost count
raining flooding Florida Tampa Bay Tampa 18 or 19 days. I've lost count
Named Entities: ['Florida Tampa Bay']
```

Example: output of dataset_parser.py:
```
Starting...
Parsing dataset...
Done parsing, dataset length: 10876
Splitting into train 0.9 and test 0.1
Generating corpuses and labels...
Test unigrams:
Generating bag of words...
Fitting...
Measure times for function: fit_forest (2017-07-17 20:49:03)
Total running time of fit_forest in seconds: 32
Measure times for function: fit_naive_bayes (2017-07-17 20:49:36)
Total running time of fit_naive_bayes in seconds: 0
Predicting...
FOREST:
acc: 0.801470588235
NAIVE BAYES:
acc: 0.827205882353
Test unigrams and bigrams:
Generating bag of words...
Fitting...
Measure times for function: fit_forest (2017-07-17 20:49:37)
Total running time of fit_forest in seconds: 88
Measure times for function: fit_naive_bayes (2017-07-17 20:51:06)
Total running time of fit_naive_bayes in seconds: 0
Predicting...
FOREST:
acc: 0.801470588235
NAIVE BAYES:
acc: 0.791360294118
```

Example: output of dataset_parser.py:
```
=== Total 2273 relevant tweets found, for example:
	Since the chemical-weapons 'red line' warning on 20 August 2012 LCC have confirmed that at least 96355 people have been killed in #Syria.
		 Tags in tweet:['Syria']
		 Users in tweet:[]
		 Urls in tweet:[]

	#Bestnaijamade: 16yr old PKK suicide bomber who detonated bomb in ... http://t.co/KSAwlYuX02 bestnaijamade bestnaijamade bestnaijamade be??_
		 Tags in tweet:[u'Bestnaijamade']
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/KSAwlYuX02']
		 Following url: http://t.co/KSAwlYuX02 -> http://bestnaijamade.blogspot.com/2015/08/16yr-old-pkk-suicide-bomber-who.html?spref=tw -> http://bestnaijamade.blogspot.co.il/2015/08/16yr-old-pkk-suicide-bomber-who.html?spref=tw

	3 Former Executives to Be Prosecuted in Fukushima Nuclear Disaster http://t.co/JSsmMLNaQ7
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/JSsmMLNaQ7']
		 Following url: http://t.co/JSsmMLNaQ7 -> http://nyti.ms/1Eatv40 -> http://www.nytimes.com/glogin?URI=http://www.nytimes.com/2015/08/01/world/asia/3-former-tepco-executives-to-be-prosecuted-in-fukushima-nuclear-disaster.html%3Futm_content%3Dbuffer7c8c6%26utm_medium%3Dsocial%26utm_source%3Dtwitter.com%26utm_campaign%3Dbuffer%26_r%3D0&utm_content=buffer41e70&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer -> http://www.nytimes.com/2015/08/01/world/asia/3-former-tepco-executives-to-be-prosecuted-in-fukushima-nuclear-disaster.html?utm_content=buffer7c8c6&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer&_r=0 -> https://www.nytimes.com/2015/08/01/world/asia/3-former-tepco-executives-to-be-prosecuted-in-fukushima-nuclear-disaster.html?utm_content=buffer7c8c6&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer&_r=0

	Toddler drowned in bath after mum left room to fetch his pyjamas http://t.co/k9aSKtwXfL
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/k9aSKtwXfL']
		 Following url: http://t.co/k9aSKtwXfL - Timeout

	Yesterday's #hailstorm! #boston #cambridge http://t.co/HbgYpruvO7 http://t.co/SwtgHLibs2
		 Tags in tweet:[u'hailstorm', u'boston', u'cambridge']
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/HbgYpruvO7', 'http://t.co/SwtgHLibs2']
		 Following url: http://t.co/HbgYpruvO7 -> http://ift.tt/1IqdZTx -> https://instagram.com/p/6A9-puHvEO/ -> https://www.instagram.com/p/6A9-puHvEO/
		 Following url: http://t.co/SwtgHLibs2 -> https://twitter.com/ItsAvanti/status/629026474679517185/photo/1


=== Total 2793 irrelevant tweets found, for example:
	#3: Car Recorder ZeroEdge?? Dual-lens Car Camera Vehicle Traffic/Driving History/Accident Camcorder  Large Re... http://t.co/kKFaSJv6Cj
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/kKFaSJv6Cj']
		 Following url: http://t.co/kKFaSJv6Cj -> http://amzn.to/1T7Omvy -> http://www.amazon.com/Simultaneous-Water-proof-Monitoring-one-button-Multi-Language/dp/B010H0SL9O/ref=pd_zg_rss_ts_e_10980561_3?ie=UTF8&tag=ocarvidplay-xd-20&utm_source=twitterfeed&utm_medium=twitter -> https://www.amazon.com/Simultaneous-Water-proof-Monitoring-one-button-Multi-Language/dp/B010H0SL9O/ref=pd_zg_rss_ts_e_10980561_3?ie=UTF8&tag=ocarvidplay-xd-20&utm_source=twitterfeed&utm_medium=twitter -> https://www.amazon.com/Z-Edge-Camera-Recorder-Vehicles-Included/dp/B010H0SL9O

	Just thought I'd let you all know...
It's probably not a good idea to plug in your hairdryer when it's wet you will be electrocuted.
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:[]

	The Eden Hazard of Hockey https://t.co/RbbnjkoqUD
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:['https://t.co/RbbnjkoqUD']
		 Following url: https://t.co/RbbnjkoqUD -> https://vine.co/v/eHzipDIO5x7

	@Luzukokoti it's all  about understanding umntu wakho. If you do and trust your partner then y OK u will know and won't fear to do anything.
		 Tags in tweet:[]
		 Users in tweet:['Luzukokoti']
		 Urls in tweet:[]

	My favorite text http://t.co/5U5GAkX2ch
		 Tags in tweet:[]
		 Users in tweet:[]
		 Urls in tweet:['http://t.co/5U5GAkX2ch']
		 Following url: http://t.co/5U5GAkX2ch -> https://twitter.com/__hailstorm/status/629117562047959041/photo/1
```
