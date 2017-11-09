import os
import argparse
import pandas

def parse_args():
	parser = argparse.ArgumentParser();
	parser.add_argument('tweets_dir', type=str, help='The directory containing the tweets')
        parser.add_argument('output_dir', type=str, help='The output directory for the combined files')
	return parser.parse_args()

def combine(outputFile, filenames):
	dfs = []
	for file in filenames:
		print("Reading "+file)
		dfs.append(pandas.read_csv(file, delimiter=';', quotechar='"'))
	pandas.concat(dfs).to_csv(outputFile)

def get_files_by_incident(tweetsDir):
	for path, dirnames, filenames in os.walk(tweetsDir):
		incidents = set([name.split('-')[0] for name in filenames])
		filesByIncident = {incident: [] for incident in incidents}
		for file in filenames:
			incident = file.split('-')[0]
			filesByIncident[incident].append(os.path.join(path, file))
		return filesByIncident

def main():
	args = parse_args()
	filesByIncident = get_files_by_incident(args.tweets_dir)
	for incident in filesByIncident:
		outputFile = os.path.join(args.output_dir, incident+'.csv')
		print("Combining " + str(len(filesByIncident[incident])) + " files into " + outputFile)
		combine(outputFile, filesByIncident[incident])


if __name__ == "__main__":
	main()
