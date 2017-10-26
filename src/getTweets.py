import argparse
import csv
import datetime
import subprocess


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('csv_path', type=str,
						 help='the path to a csv containing metadata')
	return parser.parse_args()

def make_dict(csv_path):
	shootingDict = {}
	with open(csv_path) as f:
		reader = csv.reader(f)
		header = next(reader)
		for line in reader:
			rawDate = line[0]
			date, incidentID = make_date_incident(rawDate)
			if line[1] != "":
				hashtags = line[1].split(', ')
			else:
				hashtags = []
			twitter = line[2].lower() == "true"
			shooter = line[3]
			race = line[4].lower()
			# Add to dict
			if incidentID not in shootingDict:
				shootingDict[incidentID] = {
					"date": date,
					"hashtags": hashtags,
					"shooter": shooter,
					"race": race,
					"twitter": twitter
				}
	return shootingDict


def make_date_incident(date):
	date = date.split("/")
	month = int(date[0])
	day = int(date[1])
	year = int(date[2])
	if year < 18:
		year = int("20" + date[2])
	else:
		year = int("19" + date[2])
	date = datetime.date(year, month, day)
	if day < 10:
		day = ("0" + str(day))	
	incidentID = int(str(year) + str(month) + str(day))
	return date, incidentID

def get_tweets(shootingDict):
	for incident in shootingDict:
		if shootingDict[incident]["twitter"]:
			queries = shootingDict[incident]["hashtags"]
			queries.append(shootingDict[incident]["shooter"])
			for query in queries:  # Go through the list of search terms
				currentDate = shootingDict[incident]['date']
				#for i in range(0,7):  # Go through the seven day range following incident
				for i in range(0,2):  # to run an abbreviated version for testing purposes
					nextDay = currentDate + datetime.timedelta(days=1)
					filename = "../data/tweets/" + str(incident) + "-" + query + "-" + str(i) + ".csv"
					command = "python ../GetOldTweets-python-master/Exporter.py \
							--maxtweets 10000 --querysearch '{}' --output '{}' --since {} --until {}". \
							format(query, filename, str(currentDate), str(nextDay))
					subprocess.check_output(command, shell = True)
					currentDate = nextDay  # Increase the day by one for the next one
				
	
def main():
	args = parse_args()
	shootingDict = make_dict(args.csv_path)
	get_tweets(shootingDict)


if __name__ == "__main__":
    main()